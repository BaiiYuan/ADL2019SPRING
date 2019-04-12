import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import elmo_models
from sys import stdout
from IPython import embed
from collections import Counter
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "../data"


pad_token = 0
unk_token = 1
bos_token = 2

def load_data(args):
    print("Loading data...")
    dataset = []
    all_label = set([pad_token, unk_token, bos_token])

    with open(os.path.join(data_path, "all_sentence.pkl"), "rb") as f:
        raw = pickle.load(f)
    print("Done")

    for cou, senten in enumerate(raw):
        all_label.update(senten)
        if len(senten) < args.max_length:
            senten = [pad_token]*(args.max_length-len(senten))+ [bos_token] + senten
        else:
            senten = [bos_token] + senten[:args.max_length]
        assert(len(senten) == args.max_length+1)
        dataset.append(senten)
        stdout.write(f"\r{cou}/{len(raw)}")

    dictid2idx = {dictid:idx for idx,dictid in enumerate(all_label)}

    return dataset, dictid2idx

def data_generator(args, data, batch_size_origin, dictid2idx, shuffle=True):
    if shuffle:
        used_data = random.sample(data, len(data))
    else:
        used_data = np.copy(data)

    batch_size = batch_size_origin
    num_data = len(used_data)

    global steps_per_epoch
    steps_per_epoch = num_data // batch_size

    for i in range(steps_per_epoch):
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_data = np.array(used_data[start:end])

        input_data = batch_data[:, :args.max_length]
        output_data = batch_data[:, -args.max_length:]

        output_data = output_data.tolist()
        output_data = [[dictid2idx[i] for i in line] for line in output_data]

        input_data = torch.tensor(input_data, dtype=torch.long)
        output_data = torch.tensor(output_data, dtype=torch.long)

        yield input_data.to(device), output_data.to(device)

def train(args, epoch, dataset, objective, dictid2idx):
    print("------------------------------------------")
    gen = data_generator(args, dataset, args.batch_size, dictid2idx, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []

    for  idx, (input_data, output_data) in enumerate(gen):
        # Forward and backward.
        optimizer.zero_grad()

        pred = model(input_data)
        if args.attn == 1 or args.attn == 3:
            pred = pred[0]
        loss = objective(pred, labels)

        loss.backward()
        optimizer.step()

        loss = loss.data.cpu().item()
        acc = (nn.Sigmoid()(pred).round() == labels).float().cpu().tolist()
        epoch_loss.append(loss)
        epoch_acc.append(acc)

        Iter = 100.0 * (idx + 1) / steps_per_epoch
        stdout.write("\rEpoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}".format(epoch,
                                                                                       args.epochs,
                                                                                       Iter,
                                                                                       np.mean(epoch_loss),
                                                                                       np.mean(epoch_acc)))
        if (idx + 1) % args.print_iter == 0 :
            print(" ")

    print("\n> Spends {:.2f} seconds.".format(time.time() - t1))
    print("> The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100,
                                                                              np.mean(epoch_loss)))

def trainIter(args):
    dataset, objective, word2dictid, dictid2idx, max_recall = trainInit(args)
    print(max_recall)
    for epoch in range(args.epochs):
        train(args, epoch+1, dataset, objective, dictid2idx)
        with torch.no_grad():
            model.eval()
            pass

        # save_model(args, epoch, max_recall)

def trainInit(args):
    max_recall = 0
    dataset, dictid2idx = load_data(args)

    word2dictid, vectors = create_model(args, dictid2idx)
    dictid2word = {b:a for a,b in word2dictid.items()}

    if args.model_load != None:
        print("> Loading trained model and Train")
        load_model(args.model_load)


    objective = nn.NLLLoss().to(device)

    return dataset, objective, word2dictid, dictid2idx, max_recall

def create_model(args, dictid2idx):
    print("> Create model.")
    with open(os.path.join(args.data_path, "word2idx-vectors.pkl"), "rb") as f:
        [word2dictid, vectors] = pickle.load(f)

    global model
    model = elmo_models.elmo_model(hidden_size=args.hidden_size,
                                   drop_p=args.drop_p,
                                   num_of_words=len(word2dictid),
                                   out_of_words=len(dictid2idx)
                                )

    model.word_embedding.load_state_dict({'weight': vectors.to(torch.float32)})
    model.word_embedding.weight.requires_grad = False

    model = model.to(device)
    print(model)

    global optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)

    return word2dictid, vectors

def main(args):
    trainIter(args)


if __name__ == '__main__':
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='../data')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-hn', '--hidden_size', type=int, default=150)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-dr', '--drop_p', type=float, default=0.2)
    parser.add_argument('-md', '--model_dump', type=str, default='./elmo_model_ver1.tar')
    parser.add_argument('-ml', '--model_load', type=str, default=None, help='Print every p iterations')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('--max_length', type=int, default=64, help='Max sequence length')
    args = parser.parse_args()
    main(args)