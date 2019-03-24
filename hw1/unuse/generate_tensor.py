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


data_path = "gensim_300"
word_model = Word2Vec.load("Word2Vec_V1.h5")
vectors = word_model.wv
max_length = (112, 16)

def get_tensor(tmp1, tmp2):
    if len(tmp1) > max_length[0]:
        tmp1 = tmp1[-max_length[0]:]
    tmp1 = np.array([vectors[i] for i in tmp1])
    if tmp1.shape[0] < max_length[0]:
        tmp1 = np.concatenate((tmp1, np.zeros((max_length[0]-tmp1.shape[0], 300))), axis=0)
    assert tmp1.shape[0] == max_length[0]

    if len(tmp2) > max_length:
        tmp2 = tmp2[-max_length:]
    tmp2 = np.array([vectors[i] for i in tmp2])
    if tmp2.shape[0] < max_length:
        tmp2 = np.concatenate((tmp2, np.zeros((max_length-tmp2.shape[0], 300))), axis=0)
    assert tmp2.shape[0] == max_length[1]
    return tmp


def generate_train():
    with open(os.path.join(data_path, "train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    print("Training data finish")

    write = []
    for cou in range(len(train_data)):
        data = train_data[cou]
        records = data['records']
        records = (" ".join([" ".join(i) for i in records])).split()

        thisID = data['thisID']
        wa = data['wrong_answer']
        wa = random.sample(wa, 3)
        ca = data['correct_answer']

        write.append(get_tensor(records, ca))
        for item in wa:
            write.append(get_tensor(records, item))

        if (cou+1)%2500 == 0:
            name = 'embed_data/train-{}.pt'.format(int((cou+1)/2500))
            print(name)
            write = torch.tensor(write, dtype=torch.float32)
            torch.save(write, name)
            write = []

def generate_valid():
    with open(os.path.join(data_path, "valid.pkl"), "rb") as f:
        valid_data = pickle.load(f)
    for cou in range(len(valid_data)):
        records = valid_data[cou]['records']
        valid_data[cou]['records'] = (" ".join([" ".join(i) for i in records])).split()

    print("Validation data finish")
    write = []
    for cou, data in enumerate(valid_data):

        thisID = data['thisID']
        records = data['records']
        wa = data['wrong_answer']
        ca = data['correct_answer']

        out = []
        out.append(get_tensor(records+ca))
        for item in wa:
            out.append(get_tensor(records+item))
        write.append(out)
        if (cou+1)%100 == 0:
            name = 'embed_data/valid-{}.pt'.format(int((cou+1)/100))
            print(name)
            write = torch.tensor(write, dtype=torch.float)
            torch.save(write, name)
            write = []


def generate_test():
    with open(os.path.join(data_path, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    for cou in range(len(test_data)):
        records = test_data[cou]['records']
        test_data[cou]['records'] = (" ".join([" ".join(i) for i in records])).split()

    print("Testing data finish")
    write = []
    for cou, data in enumerate(test_data):

        thisID = data['thisID']
        records = data['records']
        wa = data['wrong_answer']

        out = []
        for item in wa:
            out.append(get_tensor(records+item))
        write.append(out)
        if (cou+1)%100 == 0:
            name = 'embed_data/test-{}.pt'.format(int((cou+1)/100))
            print(name)
            write = torch.tensor(write, dtype=torch.float)
            torch.save(write, name)
            write = []


def load():
    # generate_valid()
    # generate_test()
    generate_train()
    embed()

def main():
    load()

if __name__ == '__main__':
    main()