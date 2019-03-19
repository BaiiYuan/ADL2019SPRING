import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time
import random
import pickle
import argparse
import numpy as np

import models
from sys import stdout
from IPython import embed
from collections import Counter
from gensim.models import Word2Vec

device = "cuda" if torch.cuda.is_available else "cpu"

Draw = []    

def get_embedding(args, arr, vectors, mean_vector):
    if len(arr) > args.max_length:
        arr = arr[-args.max_length:]
    arr = [vectors[i] for i in arr]
    else if len(arr) < args.max_length:
        arr = arr + [np.zeros(300) for _ in range(args.max_length-len(arr))]
    assert(len(arr)==args.max_length)
    return arr

def calculateRecall(datas, model, vectors, mean_vector, at=10):
    model.eval()
    recall = []
    for uni_data in datas:
        input_data = []
        records = uni_data['records']
        wa = uni_data['wrong_answer']
        ca = uni_data['correct_answer']
        input_data.append(get_embedding(args, records+ca, vectors, mean_vector))
        for i in range(99):
            input_data.append(get_embedding(args, records+wa[i], vectors, mean_vector))
        input_data = torch.tensor(input_data, dtype=torch.float)
        pred = model(input_data.to(device))
        embed()

def random_sample(dataset, args, rate=3):
    out = []
    for i in dataset:
        records = i['records']
        wa = i['wrong_answer']
        wa_sam = random.sample(wa, rate)
        ca = i['correct_answer']
        out.append([records+ca, 1])
        for item in wa_sam:
            out.append([records+item, 0])
    return out


def load_data(args):
    print("Load prepro-data...")
    dataset = {}
    with open(os.path.join(args.data_path, "train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.data_path, "valid.pkl"), "rb") as f:
        valid_data = pickle.load(f)
    with open(os.path.join(args.data_path, "test.pkl"), "rb") as f:
        dataset['test'] = pickle.load(f)
    print("Finish Loading!")

    for cou in range(len(train_data)):
        records = train_data[cou]['records']
        train_data[cou]['records'] = (" ".join([" ".join(i) for i in records])).split()
    dataset['train'] = random_sample(train_data, args)
    dataset['train'] = cutMaxLength(args, dataset['train'])

    # for cou in range(len(valid_data)):
    #     records = valid_data[cou]['records']
    #     valid_data[cou]['records'] = (" ".join([" ".join(i) for i in records])).split()
    # dataset['valid'] = valid_data

    return dataset

def create_model(argsargs):
    print("Create model.")
    model = models.RNNbase(window_size=args.max_length)
    # model = models.RNNatt()

    print(model)
    return model.to(device)

def data_generator(args, data, batch_size_origin, vectors, mean_vector, shuffle=True):
    if shuffle:
        used_data = random.sample(data, len(data))
    else:
        used_data = np.copy(data)

    batch_size = batch_size_origin
    num_data = len(used_data)

    global steps_per_epoch
    steps_per_epoch = num_data // batch_size # if (num_data%batch_size)==0 else (num_data // batch_size) +1

    for i in range(steps_per_epoch):
        start = i * batch_size
        end = (i + 1) * batch_size 

        batch_data = used_data[start:end]
        input_data, labels = zip(*batch_data)
        input_data = [get_embedding(args, item, vectors, mean_vector) for item in input_data]

        input_data = torch.tensor(input_data, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        yield input_data.to(device), labels.to(device)

def train(args, epoch, dataset, model, optimization, objective, vectors, mean_vector):
    # TODO: prepare training data and validation
    gen = data_generator(args, dataset['train'], args.batch_size, vectors, mean_vector, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    model.train(True)

    for  idx, (input_data, labels) in enumerate(gen):
        # Forward and backward.
        optimization.zero_grad()
        # embed()
        pred = model(input_data)
        loss = objective(pred, labels)

        loss.backward()
        optimization.step()

        # clip = 50.0
        # _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

        loss = loss.data.cpu().item()
        acc = (nn.Sigmoid()(pred).round() == labels).float().cpu().tolist() 
        epoch_loss.append(loss)
        epoch_acc.append(acc)

        Iter = 100.0 * (idx + 1) / steps_per_epoch
        stdout.write("\rEpoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}".format(epoch, args.epochs, Iter, np.mean(epoch_loss), np.mean(epoch_acc)))
        if (idx + 1) % args.print_iter == 0 :
            print(" ")

    print(" Spends {:.2f} seconds.".format(time.time() - t1))
    print("The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100, np.mean(epoch_loss)))

def load_embedding():
    word_model = Word2Vec.load("Word2Vec_V1.h5")
    vectors = word_model.wv
    mean_vector = vectors.vectors.mean(axis=0)
    return vectors, mean_vector

def trainInit(args):
    vectors, mean_vector = load_embedding()
    model = create_model(args)
    optimization = optim.Adam(model.parameters(), lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3.])).to(device)
    dataset = load_data(args)
    return dataset, model, optimization, objective, vectors, mean_vector

def trainIter(args):
    dataset, model, optimization, objective, vectors, mean_vector = trainInit(args)
    # calculateRecall(dataset['valid'], model, vectors, mean_vector)
    for epoch in range(args.epochs):
        train(args, epoch+1, dataset, model, optimization, objective, vectors, mean_vector)

def main(args):
    # init
    global loss_epoch_tr
    global acc_epoch_tr
    global loss_epoch_va
    global acc_epoch_va

    loss_epoch_tr = []
    acc_epoch_tr = []
    loss_epoch_va = []
    acc_epoch_va = []

    if args.train:
        trainIter(args)

if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./gensim_300')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-md', '--model_dump', type=str, default='./model.tar')
    parser.add_argument('-o', '--output_csv', type=str, default='output.csv')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('-s', '--save_iter', type=int, default=30, help='Save every p iterations')
    parser.add_argument('-ml', '--max_length', type=int, default=200, help='Max dialogue length')
    parser.add_argument('-tr', '--train', type=int, default=1, help='Train and test: 1, Only test: 0')
    args = parser.parse_args()

    main(args)