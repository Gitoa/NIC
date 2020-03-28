import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from data_loader import get_loader 
from build_vocab import Vocabulary
from build_split_vector import ImageTokenVector
from torchvision import transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from rcnn import FasterRCNN
import torch.optim as optim
import time


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    """
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    """
    transform = transforms.Compose([transforms.RandomCrop(args.crop_size), transforms.ToTensor()])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(args.imgtoken2vec_path, 'rb') as f:
        imgtoken2vec = pickle.load(f)
    freq_n = np.zeros(1308)
    freq_v = np.zeros(85)
    freq_j = np.zeros(96)
    for k, v in imgtoken2vec.img2vec_n.items():
            freq_n += v
            freq_v += imgtoken2vec(k)['V']
            freq_j += imgtoken2vec(k)['J']

    torch.set_printoptions(precision=2)
    print(len(freq_n))
    print(np.max(freq_n))
    print(np.max(freq_v))
    print(np.max(freq_j))
    print("len(vocab):", len(vocab))
    freq_n[freq_n<1] = 82783
    freq_j[freq_j<1] = 82783
    freq_v[freq_v<1] = 82783
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             imgtoken2vec, transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    #backbone = resnet_fpn_backbone('resnet50', True)
    num_classes = {
        'V': 85,
        'N': 1308,
        'J': 96
    }
    fasterrcnn_resnet50_fpn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    rpn = fasterrcnn_resnet50_fpn.rpn
    rpn.training = False
    backbone = fasterrcnn_resnet50_fpn.backbone
    backbone.training = False
    rpn._post_nms_top_n = {'training': 100, 'testing': 100}
    resnet50 = torchvision.models.resnet50(pretrained=True)
    modules = list(resnet50.children())[:-1]
    backbone_classification = nn.Sequential(*modules)
    for p in backbone_classification.parameters():
        p.requires_grad = False
    model = FasterRCNN(backbone, backbone_classification=backbone_classification, c_out_channels=resnet50.fc.in_features, rpn=rpn, num_classes=num_classes, freq_n=freq_n, freq_j=freq_j, freq_v=freq_v)
    PATH = '../models-BCEWithLogitsLoss/multilabel_roi_head-epoch-80.pt'
    #model.roi_heads.load_state_dict(torch.load(PATH))
    model.to(device)
    optimizer = optim.Adam(model.roi_heads.parameters())
    # Loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    #params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    #optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    loss_history = torch.Tensor([]).to(device)
    start = time.time()
    print('total_step: ', total_step)
    for epoch in range(args.num_epochs):
        loss_epoch = torch.Tensor([]).to(device)
        start_time = time.time()
        for i, (images, targets) in enumerate(data_loader):
            
            images = images.to(device)
            print(images.size())

            # Forward, backward and optimize
            losses, probs = model(images, targets)
            #print(losses)
            loss_epoch = torch.cat((loss_epoch, losses['loss_classifiers']))
            model.zero_grad()
            losses['loss_classifiers'].backward()
            optimizer.step()          
            # Forward, backward and optimize
            """
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            """
            # Print log info
            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(epoch + 1, args.num_epochs, i + 1, total_step, losses['loss_classifiers'].item()))
                print('')
                for key, v in probs.items():
                    print(key)
                    prob_group = probs[key]
                    length = len(prob_group)
                    labels = targets[key]
                    for i in range(3):
                        label = labels[i]
                        prob = prob_group[i]
                        label = label.data.numpy()
                        ori_ind = np.where(label==1)[0]
                        s_arr, ind = prob.sort(descending=True)
                        ind = ind.data.cpu().numpy()
                        pos = [np.argwhere(ind==k)[0][0]+1 for k in ori_ind]
                        print(np.sort(pos))
                print('')
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(model.roi_heads.state_dict(), os.path.join(args.model_path, 'multilabel_roi_head-{}-{}.pt'.format(epoch+1, i+1)))
        end = time.time()
        #if epoch % 3 == 1:
        torch.save(model.roi_heads.state_dict(), os.path.join(args.model_path, 'multilabel_roi_head-epoch-{}.pt'.format(epoch+1)))
        #save loss history
        mean_loss = loss_epoch.mean().reshape(1)
        loss_history = torch.cat((loss_history, mean_loss))
        print('Epoch [{}/{}], Total time: {:.4f}s, Epoch time: {:.4f}s, Mean loss: {}'.format(epoch + 1, args.num_epochs, end - start, end - start_time, loss_epoch.mean()))
        print('')
    with open(args.loss_history_path, 'wb') as f:
        pickle.dump(loss_history, f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models-BCEWithLogitsLoss-coco/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../data/coco/images/resized_train2014', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='../../data/coco/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=2000, help='step size for saving trained models')
    parser.add_argument('--imgtoken2vec_path', type=str, default='../data/img2vec_n_v_j.pkl', help='path for imgtoken2vec wrapper')
    parser.add_argument('--loss_history_path', type=str, default='./loss_history_BCE-coco.pkl', help='path for saving the loss history')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    print('! ! !')
    print('! ! !')
    print('Check the checkpoint path!!! all the data path!!!')
    main(args)