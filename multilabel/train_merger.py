import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from decoder_data_loader import get_loader 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from build_vocab import Vocabulary
from build_split_vector import ImageTokenVector
from torchvision import transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from rcnn import FasterRCNN
from model import DecoderRNN
from model import Merger
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
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    #backbone = resnet_fpn_backbone('resnet50', True)
    num_classes = {
        'V': 85,
        'N': 1308,
        'J': 96
    }
    attr_size = 1489

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
    
    encoder = FasterRCNN(backbone, backbone_classification=backbone_classification, c_out_channels=resnet50.fc.in_features, rpn=rpn, num_classes=num_classes, freq_n=freq_n, freq_j=freq_j, freq_v=freq_v)
    decoder = DecoderRNN(attr_size, args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    PATH = '../models-BCEWithLogitsLoss-coco/multilabel_roi_head-epoch-80.pt'
    DECODER_PATH = '../models-decoder/decoder-epoch-33.pt'

    encoder.roi_heads.load_state_dict(torch.load(PATH))
    encoder.roi_heads.training = False
    encoder.training = False
    encoder.to(device)

    decoder.load_state_dict(torch.load(DECODER_PATH))
    merger = Merger(decoder, args.hidden_size, attr_size, 512, len(vocab)).to(device)
    
    #params = list(decoder.parameters()) + list(merger.parameters())
    params = list(merger.parameters())
    optimizer = optim.Adam(params)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            images = images.to(device)
            captions = captions.to(device)
            targets = captions
            # Forward, backward and optimize
            with torch.no_grad():
                result = encoder(images)
            outputs = merger(result['embedded_features'], captions, lengths)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, targets)
            merger.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(epoch + 1, args.num_epochs, i + 1, total_step, loss))
                print('')
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(merger.state_dict(), os.path.join(args.model_path, 'merger-{}-{}.pt'.format(epoch+1, i+1)))
        end = time.time()
        #if epoch % 3 == 1:
        torch.save(merger.state_dict(), os.path.join(args.model_path, 'merger-epoch-{}.pt'.format(epoch+1)))
        print('Epoch [{}/{}], Total time: {:.4f}s, Epoch time: {:.4f}s'.format(epoch + 1, args.num_epochs, end - start, end - start_time))
        print('')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models-merger/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../data/coco/images/resized_train2014', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='../../data/coco/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=2000, help='step size for saving trained models')
    parser.add_argument('--imgtoken2vec_path', type=str, default='../data/img2vec_n_v_j.pkl', help='path for imgtoken2vec wrapper')
    parser.add_argument('--loss_history_path', type=str, default='./loss_history_decoder.pkl', help='path for saving the loss history')
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