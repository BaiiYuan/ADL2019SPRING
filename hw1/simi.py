import torch
import torch.nn as nn
import torch.optim as optim

import io
import os
import sys
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import models
from sys import stdout
from IPython import embed
from collections import Counter
from gensim.models import Word2Vec

data_path = "./gensim_1000"

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def process_each_data(data, vectors):
    thisID = data['thisID']
    records = data['records']
    wa = data['wrong_answer']
    output = []

    records = (" ".join([" ".join(i) for i in records])).split()
    record_mean = np.array([vectors[i] for i in records]).mean(axis=0)

    for item in wa:
        item_mean = np.array([vectors[i] for i in item]).mean(axis=0)
        output.append(record_mean.dot(item_mean))

    output = np.array(output).argsort()[-10:]
    output.sort()
    output = output.tolist()
    pred = "".join(["1-" if i in output else "0-" for i in range(100)])
    return thisID, pred

def main():
    with open(os.path.join(data_path, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)

    word_model_V1 = Word2Vec.load("Word2Vec_V1.h5")
    vectors1 = word_model_V1.wv
    word_model_V2 = Word2Vec.load("Word2Vec_V2.h5")
    vectors2 = word_model_V2.wv

    crawl = load_vectors("./crawl-300d-2M.vec")
    vectors = crawl

    embed()
    output = []
    for i in test_data:
        output.append(process_each_data(i, vectors2))

    df = pd.DataFrame(output, columns=['Id', 'Predict'])
    df.to_csv("cosine_gensim_1000.csv", index=None)
    embed()

if __name__ == '__main__':
    main()