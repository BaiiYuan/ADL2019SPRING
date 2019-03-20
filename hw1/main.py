import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

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
from tqdm import tqdm

from data import DatasetWithoutLabel

# training_set = DatasetWithoutLabel(["train-{}".format(i) for i in range(1, 41)])
# training_generator = data.DataLoader(training_set)

validation_set = DatasetWithoutLabel(["valid-{}".format(i) for i in range(1, 51)])
validation_generator = data.DataLoader(validation_set)

device = "cuda" if torch.cuda.is_available else "cpu"

Draw = []

# def get_embedding(max_length, arr, vectors, mean_vector, rec=True):
#     if len(arr) > max_length:
#         if rec:
#             arr = arr[-max_length:]
#         else:
#             arr = arr[:max_length]
#     arr = [vectors[i] for i in arr]
#     if len(arr) < max_length:
#         arr = arr + [np.zeros(300) for _ in range(max_length-len(arr))]
#     assert(len(arr)==max_length)
#     return arr

def tokenize(max_length, arr, word2idx, rec=True):
    if len(arr) > max_length:
        if rec:
            arr = arr[-max_length:]
        else:
            arr = arr[:max_length]
    arr = [word2idx.get(i, 0) for i in arr]
    if len(arr) < max_length:
        arr = arr + [0 for _ in range(max_length-len(arr))]
    assert(len(arr)==max_length)
    return arr

def calculateRecall(model, vectors, mean_vector, at=10):
    model.eval()
    recall = []
    print("Calculating Recall ...")
    for local_batch in validation_generator:
        for cou in range(100):
            input_data = local_batch[0][cou]
            pred = model(input_data.to(device))
            pred = nn.Sigmoid()(pred)
            out = pred.detach().cpu().numpy().argsort()[-at:][::-1].tolist()
            if 0 in out:
                recall.append(1)
            else:
                recall.append(0)
    return np.mean(recall)

def random_sample(dataset, args, word2idx, rate=3):
    out = []
    for i in dataset:
        records = i['records']
        records = tokenize(112, records, word2idx)
        ca = i['correct_answer']
        ca = tokenize(16, ca, word2idx)
        out.append([records, ca, 1])

        wa = i['wrong_answer']
        wa_sam = random.sample(wa, rate)
        for item in wa_sam:
            item = tokenize(16, item, word2idx)
            out.append([records, item, 0])
    embed()
    return out

def load_data(args, word2idx):
    print("Load prepro-data...")
    dataset = {}

    with open(os.path.join(args.data_path, "train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    for cou in range(len(train_data)):
        records = train_data[cou]['records']
        train_data[cou]['records'] = (" ".join([" ".join(i) for i in records])).split()
    dataset['train'] = random_sample(train_data, args, word2idx)
    print("Training data finish")

    return dataset



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
        input_data_rec, input_data_rep, labels = zip(*batch_data)
        embed()
        rep_len = args.max_length//8
        rec_len = args.max_length-rep_len
        input_data_rec = [get_embedding(rec_len, item, vectors, mean_vector) for item in input_data_rec]
        input_data_rep = [get_embedding(rep_len, item, vectors, mean_vector, rec=False) for item in input_data_rep]

        input_data_rec = torch.tensor(input_data_rec, dtype=torch.int32)
        input_data_rep = torch.tensor(input_data_rep, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.float32)

        yield input_data_rec.to(device), input_data_rep.to(device), labels.to(device)

def old_train(args, epoch, dataset, objective):
    # TODO: prepare training data and validation
    gen = data_generator(args, dataset['train'], args.batch_size, vectors, mean_vector, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    model.train(True)

    for  idx, (input_data_rec, input_data_rep, labels) in enumerate(gen):
        # Forward and backward.
        optimizer.zero_grad()
        embed()
        pred = model(input_data_rec, input_data_rep)
        loss = objective(pred, labels)

        loss.backward()
        optimizer.step()

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
    print("> The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100, np.mean(epoch_loss)))

# def train(args, epoch, objective, vectors, mean_vector):
#     # TODO: prepare training data and validation

#     t1 = time.time()
#     epoch_loss = []
#     epoch_acc = []
#     model.train(True)
#     idx = 0

#     assert args.batch_size%4 == 0
#     labels = torch.tensor([1., 0., 0., 0.]*(args.batch_size//4), dtype=torch.float32).to(device)

#     steps_per_epoch = 4*100000 // args.batch_size
#     for _local_batch in training_generator:
#         # Forward and backward.
#         local_batch = _local_batch[0].to(device)
#         for cou in range(local_batch.shape[0]//args.batch_size):

#             start = cou * args.batch_size
#             end = (cou + 1) * args.batch_size
#             input_data = local_batch[start:end]

#             optimizer.zero_grad()

#             pred = model(input_data)
#             loss = objective(pred, labels)

#             loss.backward()
#             optimizer.step()

#             # clip = 50.0
#             # _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

#             loss = loss.data.cpu().item()
#             acc = (nn.Sigmoid()(pred).round() == labels).float().cpu().tolist()
#             epoch_loss.append(loss)
#             epoch_acc.append(acc)

#             Iter = 100.0 * (idx + 1) / steps_per_epoch
#             stdout.write("\rEpoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}".format(epoch, args.epochs, Iter, np.mean(epoch_loss), np.mean(epoch_acc)))
#             if (idx + 1) % args.print_iter == 0 :
#                 print(" ")
#             idx += 1

#     print("> Spends {:.2f} seconds.".format(time.time() - t1))
#     print("> The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100, np.mean(epoch_loss)))
#     return model, optimizer

def create_model(args):
    print("Create model.")
    word_model = Word2Vec.load("Word2Vec_V1.h5")
    vectors = word_model.wv
    all_words = vectors.index2word
    mean_vector = vectors.vectors.mean(axis=0)
    idx2word = {cou+1:word for cou, word in enumerate(all_words)}
    word2idx = {word:cou+1 for cou, word in enumerate(all_words)}

    global model
    model = models.RNNbase(window_size=args.max_length,
                           embedding_size=128,
                           hidden_size=64,
                           num_of_words=len(all_words)+1
                        )
    # model = models.RNNatt()

    wei = torch.tensor(vectors.vectors, dtype=torch.float)
    model.word_embedding.load_state_dict({'weight': torch.cat((torch.zeros((1, 300)), wei), dim=0)})
    model.word_embedding.weight.requires_grad = False

    model = model.to(device)
    print(model)
    return word2idx, idx2word

def trainIter(args):
    max_recall = 0
    dataset, objective, word2idx, idx2word = trainInit(args)
    for epoch in range(args.epochs):
        old_train(args, epoch+1, dataset, objective)
        # score = calculateRecall(model, vectors, mean_vector)
        # if score > max_recall:
        #     max_recall = score
        #     save_model(args, epoch)
        # print("> Validation Recall: {}".format(score))

def trainInit(args):
    word2idx, idx2word = create_model(args)
    global optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([3.])).to(device)
    dataset = load_data(args, word2idx)
    return dataset, objective, word2idx, idx2word


def save_model(args, epoch):
    print("Saving Model...")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
    }, args.model_dump)

def testAll(args):
    pass

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
    testAll(args)

if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./gensim_300')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-md', '--model_dump', type=str, default='./model.tar')
    parser.add_argument('-o', '--output_csv', type=str, default='output.csv')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('-s', '--save_iter', type=int, default=30, help='Save every p iterations')
    parser.add_argument('-ml', '--max_length', type=int, default=128, help='Max dialogue length')
    parser.add_argument('-tr', '--train', type=int, default=1, help='Train and test: 1, Only test: 0')
    args = parser.parse_args()

    main(args)