import torch
import re
import os
import pickle
from IPython import embed
from sys import stdout
import nltk

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "../data"

def load_vectors(f):
    cou = 3 # <pad> as 0
    word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2}
    vectors = []
    for line in f:
        tokens = line.rstrip().split(' ')
        if cou == 138703:
            word2idx["Baiyuan"] = cou
        else:
            word2idx[tokens[0]] = cou
        cou += 1
        stdout.write("\r{} {}".format(cou, len(word2idx)))
        vectors.append(list(map(float, tokens[1:])))
    print(len(word2idx.keys()), len(vectors))
    print(" ")
    vectors = torch.tensor(vectors)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.zeros(1, 300), vectors], dim=0)

    return word2idx, vectors

def main():
    with open(os.path.join(data_path, "GloVe/glove.840B.300d.txt")) as f:
        word2idx, vectors = load_vectors(f)

    with open(os.path.join(data_path, "word2idx-vectors.pkl"), "wb") as f:
        pickle.dump([word2idx, vectors], f)

if __name__ == '__main__':
        main()