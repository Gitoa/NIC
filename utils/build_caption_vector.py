import nltk
import pickle
import argparse
import numpy as np
from nltk.corpus import wordnet
from pycocotools.coco import COCO
from build_vocab import Vocabulary

class ImageVector(object):
    """represent image by one-hot vector"""
    def __init__(self, vocab_size):
        self.img2vec = {}
        self.vocab_size = vocab_size
    
    def add_caption(self, image_id, lemma_vec):
        if not image_id in self.img2vec:
            self.img2vec[image_id] = lemma_vec
        else:
            self.img2vec[image_id] = self.img2vec[image_id] + lemma_vec
            self.img2vec[image_id][self.img2vec[image_id]>1] = 1

    def __call__(self, image_id):
        if not image_id in self.img2vec:
            return np.zeros(self.vocab_size)
        return self.img2vec[image_id]

    def __len__(self):
        return len(self.img2vec)

class ImageTokenVector(object):
    """represent image by one-hot vector"""
    def __init__(self, vocab_size):
        self.img2vec_n = {}
        self.img2vec_j = {}
        self.img2vec_v = {}
        
    def add_caption(self, image_id, lemma_vec_n, lemma_vec_v, lemma_vec_j):
        if not image_id in self.img2vec_n:
            self.img2vec_n[image_id] = lemma_vec_n
        else:
            self.img2vec_n[image_id] = self.img2vec_n[image_id] + lemma_vec_n
            self.img2vec_n[image_id][self.img2vec_n[image_id]>1] = 1

        if not image_id in self.img2vec_j:
            self.img2vec_j[image_id] = lemma_vec_j
        else:
            self.img2vec_j[image_id] = self.img2vec_j[image_id] + lemma_vec_j
            self.img2vec_j[image_id][self.img2vec_j[image_id]>1] = 1

        if not image_id in self.img2vec_v:
            self.img2vec_v[image_id] = lemma_vec_v
        else:
            self.img2vec_v[image_id] = self.img2vec_v[image_id] + lemma_vec_v
            self.img2vec_v[image_id][self.img2vec_v[image_id]>1] = 1

    def __call__(self, image_id):
        if not image_id in self.img2vec_n:
            return None
        
        return self.img2vec[image_id]

    def __len__(self):
        return len(self.img2vec)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None

def lemmatize_sentence(sentence, tags):
    res = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for word, pos in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
        if pos in tags:
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res

def build_img_vec(json, vocab):
    coco = COCO(json)
    ids = coco.anns.keys()
    vocab_size = len(vocab)
    image2vec = ImageVector(vocab_size)
    tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    unk_id = vocab('<unk>')
    for i, id in enumerate(ids):
        base_vec = np.zeros(vocab_size)
        anns = coco.anns[id]
        caption = str(anns['caption'])
        image_id = anns['image_id']
        tokens = lemmatize_sentence(caption.lower(), tags)
        for token in tokens:
            idx = vocab(token)
            if idx != unk_id and base_vec[idx] == 0:
                base_vec[idx] = 1
            
        
        image2vec.add_caption(image_id, base_vec)
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    return image2vec

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    img2vec = build_img_vec(json=args.caption_path, vocab=vocab)
    with open(args.img2vec_path, 'wb') as f:
        pickle.dump(img2vec, f)
    print("Total image size: {}".format(len(img2vec)))
    print("Saved the vocabulary wrapper to '{}'".format(args.img2vec_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../../data/coco/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='../data/lemmatized_vocab1500.pkl', 
                        help='path for vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5, 
                        help='minimum word count threshold')
    parser.add_argument('--img2vec_path', type=str, default='../data/img2vec.pkl',
                        help='path for saving img2vec wrapper')
    args = parser.parse_args()
    main(args)
