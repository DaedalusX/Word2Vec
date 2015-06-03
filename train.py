from gensim import models
from gensim import corpora
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from nltk.corpus import brown, movie_reviews, treebank
import string
import csv
import unicodedata
import sys
from collections import Counter
import pickle
import re
import cPickle
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))

path = '/Users/salilnavgire/Downloads/irene_comments.csv'


def read_data(path):
    data = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data.append(row)
    return data[0]


def prepare_data(data):
    new_data = []
    for i, res in enumerate(data):
        try:
            s = re.sub('[^a-zA-Z0-9\n\.]', ' ', res)
            new = [x for x in s.lower().split() if x not in stoplist]
            if i % 10000 == 0:
                print i
            new_data.append(new)
        except IndexError:
            pass
    return new_data


def train_model(new_data):
    model = Word2Vec(new_data)
    return model


def save_model(fname):
    model.save(fname)


def read_model(fname):
    model = Word2Vec.load(fname)
    return model


def return_vocab(model):
    vocabs = []
    for res in model.vocab:
        vocabs.append(res)
    return vocabs


def word2vec_similar(word, model=None):
    if model is None:
        model = read_model('modelv1')

    print model.most_similar(word, topn=10)
    return model.most_similar(word, topn=10)



if __name__ == '__main__':
    '''
    data = read_data(path)
    print len(data)
    new_data = prepare_data(data)
    print len(new_data)
    model = train_model(new_data)
    print 'saving model'
    save_model('modelv1')
    '''

    model = read_model('modelv1')
    # vocabs = return_vocab(model)
    # print len(vocabs)
    word2vec_similar(word='boots')

    print 'end'
