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
    def __init__(self):
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
        
        return {'J': self.img2vec_j[image_id], 
                'V': self.img2vec_v[image_id],
                'N': self.img2vec_n[image_id]}

    def __len__(self):
        return len(self.img2vec_n)

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

def build_img_vec(json, vocab, vocab_n, vocab_j, vocab_v):
    coco = COCO(json)
    ids = coco.anns.keys()
    vocab_size = len(vocab)
    vocab_n_size = len(vocab_n)
    vocab_j_size = len(vocab_j)
    vocab_v_size = len(vocab_v)
    print('n: ', vocab_n_size)
    print('j: ', vocab_j_size)
    print('v: ', vocab_v_size)

    image2vec = ImageVector(vocab_size)
    imagetoken2vec = ImageTokenVector()
    tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    unk_id_n = vocab_n('<unk>')
    unk_id_j = vocab_j('<unk>')
    unk_id_v = vocab_v('<unk>')
    
    for i, id in enumerate(ids):
        base_vec = np.zeros(vocab_size)
        vec_n = np.zeros(vocab_n_size)
        vec_j = np.zeros(vocab_j_size)
        vec_v = np.zeros(vocab_v_size)
        anns = coco.anns[id]
        caption = str(anns['caption'])
        image_id = anns['image_id']
        tokens = lemmatize_sentence(caption.lower(), tags)
        for token in tokens:
            idx = vocab(token)
            if base_vec[idx] == 0:
                base_vec[idx] = 1
            tag = nltk.pos_tag([token])[0][1]
            if tag.startswith('N'):
                idx = vocab_n(token)
                if idx != unk_id_n and vec_n[idx] == 0:
                    vec_n[idx] = 1
            if tag.startswith('J'):
                idx = vocab_j(token)
                if idx != unk_id_j and vec_j[idx] == 0:
                    vec_j[idx] = 1
            if tag.startswith('V'):
                idx = vocab_v(token)
                if idx != unk_id_v and vec_v[idx] == 0:
                    vec_v[idx] = 1
        #image2vec.add_caption(image_id, base_vec)
        imagetoken2vec.add_caption(image_id, vec_n, vec_v, vec_j)
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    return image2vec, imagetoken2vec

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(args.vocab_n_path, 'rb') as f:
        vocab_n = pickle.load(f)
    with open(args.vocab_j_path, 'rb') as f:
        vocab_j = pickle.load(f)
    with open(args.vocab_v_path, 'rb') as f:
        vocab_v = pickle.load(f)
    img2vec, imagetoken2vec = build_img_vec(json=args.caption_path, vocab=vocab, vocab_n=vocab_n, vocab_j=vocab_j, vocab_v=vocab_v)
    #with open(args.img2vec_path, 'wb') as f:
        #pickle.dump(img2vec, f)
    with open(args.imgtoken2vec_path, 'wb') as f:
        pickle.dump(imagetoken2vec, f)
    print("Total image size: {}".format(len(img2vec)))
    print("Saved the vocabulary wrapper to '{}'".format(args.img2vec_path))
    print("Total image size: {}".format(len(imagetoken2vec)))
    print("Saved the vocabulary wrapper to '{}'".format(args.imgtoken2vec_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../../data/coco/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='../data/lemmatized_vocab1500.pkl', 
                        help='path for vocabulary wrapper')
    parser.add_argument('--vocab_n_path', type=str, default='../data/lemmatized_vocab_n.pkl',
                        help='path for vocabulary wrapper of n')
    parser.add_argument('--vocab_v_path', type=str, default='../data/lemmatized_vocab_v.pkl',
                        help='path for vocabulary wrapper of v')
    parser.add_argument('--vocab_j_path', type=str, default='../data/lemmatized_vocab_j.pkl',
                        help='path for vocabulary wrapper of j')
    parser.add_argument('--threshold', type=int, default=5, 
                        help='minimum word count threshold')
    parser.add_argument('--img2vec_path', type=str, default='../data/img2vec.pkl',
                        help='path for saving img2vec wrapper')
    parser.add_argument('--imgtoken2vec_path', type=str, default='../data/img2vec_n_v_j.pkl',
                        help='path for saving imgtoken2vec wrapper')
    args = parser.parse_args()
    main(args)
