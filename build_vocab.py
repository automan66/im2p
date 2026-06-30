import os
import numpy as np
import json 
import nltk
import pickle
import argparse
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

if __name__ == '__main__':
    paragraph_json_file = open('/home/xianglu/dataset/VG/im2p/paragraphs_v1.json').read()
    paragraph = json.loads(paragraph_json_file)

    img2paragraph = {}

    for each_img in paragraph:
        image_id = each_img['image_id']
        each_paragraph = each_img['paragraph']
        sentences = each_paragraph.split('.')
        if '' in sentences:
            sentences.remove('')
        if ' ' in sentences:
            sentences.remove(' ')
        img2paragraph[image_id] = [len(sentences), sentences]

    with open('img2paragraph', 'wb') as f:
        pickle.dump(img2paragraph, f)

    counter = Counter()
    nsents = 0
    for key, para in img2paragraph.iteritems():
        for sent in para[1]:
            tokens = nltk.tokenize.word_tokenize(sent.lower())
            counter.update(tokens)
            nsents += 1

    threshold = 5
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    cnts = [cnt for word, cnt in counter.items() if cnt >= threshold]
    cnts = [nsents]*4 + cnts

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for i, word in enumerate(words):
        vocab.add_word(word)
    #print vocab('<unk>'),vocab('a')

    cnts = np.array(cnts)*1.0
    cnts /= np.sum(cnts) # normalize to frequencies
    cnts = np.log(cnts)
    cnts -= np.max(cnts) # shift to nice numeric range

    vocab_path = 'vocab.pkl'
    bias_init = 'bias_init.txt'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    np.savetxt(bias_init, cnts)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))