import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import utils
from PIL import Image
from build_vocab import Vocabulary
from build_split_vector import ImageTokenVector
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, imgtoken2vec, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            imgtoken2vec: one-hot lemmatized vector wrapper
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.imgtoken2vec = imgtoken2vec
        self.ids = list(self.imgtoken2vec.img2vec_n.keys())
        #self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and lemmatized vector)."""
        #vocab = self.vocab
        #ann_id = self.ids[index]
        #caption = coco.anns[ann_id]['caption']
        #img_id = coco.anns[ann_id]['image_id']
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Get the one-hot lemmatized vector of image
        tokens_vec = self.imgtoken2vec(img_id)
        target_v = torch.Tensor(tokens_vec['V'])
        target_j = torch.Tensor(tokens_vec['J'])
        target_n = torch.Tensor(tokens_vec['N'])

        # Convert caption (string) to word ids.
        """
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        """
        return image, target_v, target_j, target_n

    def __len__(self):
        #return len(self.ids)

        #Test the model using 10 * batch_size images
        return 1024


def collate_fn(data):
    """The default collate_fn is just fine"""

    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    #data.sort(key=lambda x: len(x[1]), reverse=True)
    images, token_v, token_j, token_n = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # Merge tokens (from tuple of Dict[str: 1D Tensor] to Dict[str: 2D Tensor]).
    new_token = {}
    
    new_token['V'] = torch.stack(token_v, 0)
    new_token['N'] = torch.stack(token_n, 0)
    new_token['J'] = torch.stack(token_j, 0)

    return images, new_token

def get_loader(root, json, vocab, imgtoken2vec, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       imgtoken2vec=imgtoken2vec,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, targets, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # targets: List[Dict[Tensor]] (batch_size, Dict).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader