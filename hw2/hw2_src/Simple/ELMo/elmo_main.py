import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time
import ipdb
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
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "../data"


pad_token = 0
unk_token = 1
bos_token = 2

max_sent_len = 64
max_word_len = 16

def _pad_and_cut(seq, max_leng, pad, array):
    if len(seq) < max_leng:
        seq = seq + [pad]*(max_leng-len(seq))
    else:
        seq = seq[:max_leng]
    if array:
        return np.array(seq)
    return seq

def load_data(args):
    print("Loading data...")
    dataset = []
    word_count = {}
    all_label = set([pad_token, unk_token, bos_token])
    a = []
    with open(os.path.join(args.data_path, "word2idx-vectors-v2.pkl"), "rb") as f:
        [word2idx, vectors_small] = pickle.load(f)
    with open(os.path.join(args.data_path, "language_model/corpus_tokenized.txt")) as f:
        cou = 0
        for line in f:
            a.append(line.strip())
            for w in line.strip().split():
                if w not in word_count.keys():
                    word_count[w] = 0
                word_count[w] += 1
            stdout.write("\r> {}".format(cou))
            cou+=1
            if cou > 1000000:
                break

    print("\nProcessing Data ... ")
    all_word = list(word_count.keys())
    print(f"[*] Before: len -> {len(all_word)}")

    top50k = sorted(word_count.items(), key=lambda d: -d[1])[:80000-4]
    all_word = ["<pad>", "<unk>", "<sos>", "<eos>"] + [w[0] for w in top50k]
    all_num = [w[1] for w in top50k]

    print(np.sum(all_num[:20]),
          np.sum(all_num[20:200]),
          np.sum(all_num[200:1000]),
          np.sum(all_num[1000:10000]),
          np.sum(all_num[10000:]))

    word2idx = {w:idx for idx, w in enumerate(all_word)}
    # idx2word = {idx:w for idx, w in enumerate(all_word)}
    print(f"[*] After: len -> {len(all_word)}")
    del all_word, word_count, vectors_small


    char_pad = 256
    char_sos = 257
    char_eos = 258
    char_unk = 259

    sos = np.array([char_sos]+[char_pad]*(max_word_len-1))
    eos = np.array([char_eos]+[char_pad]*(max_word_len-1))
    word_pad = np.array([char_pad]*max_word_len)

    cou = 0
    for sent in a:
        stdout.write("\r> {}".format(cou))
        cou += 1
        word_list = sent.split()
        out = [ _pad_and_cut(seq=[min(ord(c), char_unk) for c in word],
                             max_leng=max_word_len,
                             pad=char_pad,
                             array=True
                             ) for word in word_list]

        pad_out = _pad_and_cut(seq=out,
                               max_leng=max_sent_len-1,
                               pad=word_pad,
                               array=False
                               )
        pad_out = [sos] + pad_out + [eos]

        target = word_list[:]
        target = [word2idx.get(word, 1) for word in target]
        pad_target = _pad_and_cut(seq=target,
                                  max_leng=max_sent_len-1,
                                  pad=0,
                                  array=False
                                  )
        pad_target = [2] + pad_target + [3]
        dataset.append((pad_out, pad_target))
        assert len(pad_out) == 65
        assert len(pad_target) == 65
    print("")
    return dataset, word2idx

def data_generator(args, data, batch_size_origin, shuffle=True):
    if shuffle:
        used_data = random.sample(data, len(data))
    else:
        used_data = np.copy(data)

    batch_size = batch_size_origin
    num_data = len(used_data)

    global steps_per_epoch
    steps_per_epoch =  num_data // batch_size if (num_data%batch_size)==0 else (num_data // batch_size) +1

    for i in range(steps_per_epoch):
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_data = np.array(used_data[start:end])

        input_data, labels = zip(*batch_data)

        input_data = torch.tensor(input_data, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.long)

        yield input_data.to(device), labels.to(device)

def valid_data_generator(args, data, batch_size_origin, shuffle=True):
    if shuffle:
        used_data = random.sample(data, len(data))
    else:
        used_data = np.copy(data)

    batch_size = batch_size_origin
    num_data = len(used_data)

    valid_step = num_data // batch_size if (num_data%batch_size)==0 else (num_data // batch_size) +1
    for i in range(valid_step):
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_data = np.array(used_data[start:end])

        input_data, labels = zip(*batch_data)

        input_data = torch.tensor(input_data, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.long)

        yield input_data.to(device), labels.to(device)

def train(args, epoch, dataset, criterion, min_loss, valid_data):
    print("------------------------------------------")
    gen = data_generator(args, dataset, args.batch_size, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    model.train(True)

    for idx, (input_data, labels) in enumerate(gen):
        # Forward and backward.
        optimizer.zero_grad()
        loss = 0

        pred = model(input_data, labels)
        # for i in range(input_data.shape[0]):
        #     loss += criterion(pred[i], labels[i])

        loss_f, loss_b = pred
        loss_f.backward()
        loss_b.backward()
        optimizer.step()

        loss_f = loss_f.data.cpu().item()/args.batch_size
        loss_b = loss_b.data.cpu().item()/args.batch_size

        epoch_loss.append((loss_f+loss_b)/2)

        train_loss.append((loss_f, loss_b))
        # acc = (pred.argmax(dim=2) == labels).float().cpu().tolist()
        # epoch_acc.extend(acc)

        Iter = 100.0 * (idx + 1) / steps_per_epoch
        stdout.write("\r> Train: IDx: {}. Epoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}".format(idx,epoch,
                                                                                                       args.epochs,
                                                                                                       Iter,
                                                                                                       np.mean(epoch_loss),
                                                                                                       0#np.mean(epoch_acc)*100
                                                                                                       ))
        if (idx + 1) % args.print_iter == 0 :
            print(" ")
            with torch.no_grad():
                model.eval()
                tmp_loss = valid(args, valid_data, criterion)
            if tmp_loss < min_loss:
                min_loss = tmp_loss
                save_model(args, epoch, min_loss)
            model.train(True)
            print("------------------------------------------")


    print("\n> Spends {:.2f} seconds.".format(time.time() - t1))
    print("[*] The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(0,#np.mean(epoch_acc)*100,
                                                                                np.mean(epoch_loss)))
    return min_loss

def valid(args, dataset, criterion):
    print("------------------------------------------")
    gen = valid_data_generator(args, dataset, args.batch_size, shuffle=False)  # generate train data
    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    tmp_f, tmp_b = [], []
    for idx, (input_data, labels) in enumerate(gen):
        loss = 0
        pred = model(input_data, labels)

        loss_f, loss_b = pred

        loss_f = loss_f.data.cpu().item()/args.batch_size
        loss_b = loss_b.data.cpu().item()/args.batch_size

        epoch_loss.append((loss_f+loss_b)/2)
        tmp_f.append(loss_f)
        tmp_b.append(loss_b)

        stdout.write("\r> Valid: Idx: {}, Loss: {:.4f}, Acc: {:.2f}%".format(idx,
                                                                             np.mean(epoch_loss),
                                                                             0#np.mean(epoch_acc)*100,
                                                                             ))
    valid_loss.append((np.mean(tmp_f), np.mean(tmp_b)))
    np.save("train_loss.npy", train_loss)
    np.save("valid_loss.npy", valid_loss)
    print("\n> Spends {:.2f} seconds.".format(time.time() - t1))
    print("[*] The Validation dataset Loss is {:.4f}".format(np.mean(epoch_loss)))
    return np.mean(epoch_loss)

def trainIter(args):
    dataset, criterion, min_loss = trainInit(args)
    train_data, valid_data = train_test_split(dataset, test_size=10000)
    print(min_loss)
    for epoch in range(args.epochs):
        min_loss = train(args, epoch+1, train_data, criterion, min_loss, valid_data)
        with torch.no_grad():
            model.eval()
            tmp_loss = valid(args, valid_data, criterion)
        if tmp_loss < min_loss:
            min_loss = tmp_loss
            save_model(args, epoch, min_loss)


def trainInit(args):
    min_loss = 66666
    dataset, word2idx = load_data(args)
    create_model(args, word2idx)

    if args.model_load != None:
        print("[*] Loading trained model and Train")
        min_loss = load_model(args.model_load)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = criterion.to(device)

    return dataset, criterion, min_loss

def create_model(args, word2idx):
    print("[*] Create model.")

    global model, high_net, char_embed
    model = elmo_models.elmo_model(input_size=args.hidden_size,
                                   hidden_size=args.hidden_size,
                                   drop_p=args.drop_p,
                                   out_of_words=len(word2idx)
                                   )

    # model.word_embedding.load_state_dict({'weight': vectors.to(torch.float32)})
    # model.word_embedding.weight.requires_grad = False

    model = model.to(device)
    print(model)

    global optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)
    return

def save_model(args, epoch, min_loss):
    print("[*] Saving Model")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'min_loss': min_loss
    }, args.model_dump)

def loadchkpt(ckptfile):
    ckpt = torch.load(ckptfile)
    model.load_state_dict(ckpt['model'])
    return ckpt['min_loss']

def main(args):
    global train_loss, valid_loss
    train_loss = []
    valid_loss = []
    trainIter(args)
    np.save("train_loss.npy", train_loss)
    np.save("valid_loss.npy", valid_loss)

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser.add_argument('-dp', '--data_path', type=str, default='../data')
        parser.add_argument('-e', '--epochs', type=int, default=2)
        parser.add_argument('-b', '--batch_size', type=int, default=256)
        parser.add_argument('-hn', '--hidden_size', type=int, default=512)
        parser.add_argument('-lr', '--lr_rate', type=float, default=1e-3)
        parser.add_argument('-dr', '--drop_p', type=float, default=0.1)
        parser.add_argument('-rv', '--reverse', type=int, default=0)
        parser.add_argument('-md', '--model_dump', type=str, default='./elmo_model_ver1.tar')
        parser.add_argument('-ml', '--model_load', type=str, default=None, help='Print every p iterations')
        parser.add_argument('-p', '--print_iter', type=int, default=400, help='Print every p iterations')
        parser.add_argument('-mc', '--max_count', type=int, default=40)
        parser.add_argument('--max_length', type=int, default=64, help='Max sequence length')
        args = parser.parse_args()
        main(args)