import nltk
import pickle
import argparse
from build_vocab import Vocabulary

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_n = Vocabulary()
    vocab_j = Vocabulary()
    vocab_v = Vocabulary()

    vocab_n.add_word('<unk>')
    vocab_j.add_word('<unk>')
    vocab_v.add_word('<unk>')
    for word in vocab.word2idx:
        pos = nltk.pos_tag([word])[0][1]
        if pos.startswith('J'):
            vocab_j.add_word(word)
        elif pos.startswith('N'):
            vocab_n.add_word(word)
        elif pos.startswith('V'):
            vocab_v.add_word(word)
    with open(args.vocab_path_j, 'wb') as f:
        pickle.dump(vocab_j, f)
    with open(args.vocab_path_n, 'wb') as f:
        pickle.dump(vocab_n, f)
    with open(args.vocab_path_v, 'wb') as f:
        pickle.dump(vocab_v, f)
    print("V size: {}".format(len(vocab_v)))
    print("N size: {}".format(len(vocab_n)))
    print("J size: {}".format(len(vocab_j)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='../data/lemmatized_vocab1500.pkl', 
                        help='path for reading vocabulary wrapper')
    parser.add_argument('--vocab_path_j', type=str, default='../data/lemmatized_vocab_j.pkl',
                        help='path for saving vocabulary wrapper of j')
    parser.add_argument('--vocab_path_n', type=str, default='../data/lemmatized_vocab_n.pkl',
                        help='path for saving vocabulary wrapper of n')
    parser.add_argument('--vocab_path_v', type=str, default='../data/lemmatized_vocab_v.pkl',
                        help='path for saving vocabulary wrapper of v')
    args = parser.parse_args()
    main(args)