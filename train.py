import csv
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import defaultdict
stoplist = set(stopwords.words('english'))

path = '/Users/salilnavgire/Downloads/irene_comments.csv'

model_name = 'modelv1'
alphabet = 'abcdefghijklmnopqrstuvwxyz'


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


def words(text): return re.findall('[a-z]+', text.lower())


def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word, NWORDS):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words, NWORDS): return set(w for w in words if w in NWORDS)


def correct(word, NWORDS):
    candidates = known([word], NWORDS) or known(edits1(word), NWORDS) or known_edits2(word, NWORDS) or [word]
    return max(candidates, key=NWORDS.get)


def word2vec_similar(word, features, model=None):
    if model is None:
        model = read_model(model_name)

    NWORDS = train(features)
    if word in features:
        print model.most_similar(word, topn=10)
        return model.most_similar(word, topn=10)
    else:
        print 'word not in dictionary, did you mean->  ', correct(word, NWORDS)
        print correct(word, NWORDS)


def find_similar(word):
    list3 = []
    for res in features:
        if re.search(word, res):
            if len(res) <= len(word)+2:
                list3.append(res)
    return list3


def word2vec_model_cache(path):
    data = read_data(path)
    new_data = prepare_data(data)
    model = train_model(new_data)
    print 'saving model'
    save_model(model_name)
    return model


if __name__ == '__main__':
    # model = word2vec_model_cache(path)

    model = read_model(model_name)
    features = return_vocab(model)

    word2vec_similar(word='boots', features=features, model=model)

    print 'end'
