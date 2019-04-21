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

import models
from sys import stdout
from IPython import embed
# from collections import Counter
from tqdm import tqdm

from load import load_data_elmo as load_data
from load import load_test_data_elmo as load_test_data
from config import rep_len, rec_len, RATE

visualize = False
device = "cuda" if torch.cuda.is_available else "cpu"

char_pad = 256
char_sos = 257
char_eos = 258
char_unk = 259
max_word_len = 16

sos = np.array([char_sos]+[char_pad]*(max_word_len-1))
eos = np.array([char_eos]+[char_pad]*(max_word_len-1))
word_pad = np.array([char_pad]*max_word_len)

model_name = "./elmo_model_adap_58000.tar"
elmo = elmo_models.elmo_model(input_size=512,
                               hidden_size=512,
                               drop_p=0.,
                               out_of_words=80000
                               )
ckpt = torch.load(model_name)
elmo.load_state_dict(ckpt['elmo'])
elmo.to(device)
elmo.eval()

def _pad_and_cut(self, seq, max_leng, pad, tensor):
    if len(seq) < max_leng:
        seq = seq + [pad]*(max_leng-len(seq))
    else:
        seq = seq[:max_leng]
    if tensor:
        return np.array(seq)
    return seq

def calculateRecall(dataset, at=10):
    datas = dataset['valid']
    recall10, recall5 = [], []
    print("> Calculating Recall ...")
    for cou, data in enumerate(datas):
        input_data_rec, input_data_rep = zip(*data)

        pred = model(input_data_rec, input_data_rep, elmo)

        if args.attn == 1 or args.attn == 3:
            pred = pred[0]
        pred = nn.Sigmoid()(pred)
        out = pred.detach().cpu().numpy().argsort()[-at:][::-1].tolist()

        if 0 in out:    recall10.append(1)
        else:           recall10.append(0)
        if 0 in out[:5]:    recall5.append(1)
        else:               recall5.append(0)

    return np.mean(recall10), np.mean(recall5)


def data_generator(args, data, batch_size_origin, shuffle=True):
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

        batch_data = used_data[start:end]
        input_data_rec, input_data_rep, labels = zip(*batch_data)

        # input_data_rec = torch.tensor(input_data_rec, dtype=torch.long)
        # input_data_rep = torch.tensor(input_data_rep, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32)

        # yield input_data_rec.to(device), input_data_rep.to(device), labels.to(device)
        yield input_data_rec, input_data_rep, labels.to(device)

def train(args, epoch, dataset, objective):
    print("------------------------------------------")
    gen = data_generator(args, dataset['train'], args.batch_size, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    model.train(True)

    for  idx, (input_data_rec, input_data_rep, labels) in enumerate(gen):
        optimizer.zero_grad()

        pred = model(input_data_rec, input_data_rep, elmo)
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

def trainInit(args):
    max_recall = 0
    word2idx, vectors = create_model(args)
    idx2word = {b:a for a,b in word2idx.items()}

    if args.model_load != None:
        print("> Loading trained model and Train")
        max_recall = load_model(args.model_load)

    dataset = load_data(args, word2idx, vectors)
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([RATE])).to(device)

    return dataset, objective, word2idx, max_recall

def trainIter(args):
    max_recall = 0
    dataset, objective, word2idx, max_recall = trainInit(args)
    print(max_recall)
    for epoch in range(args.epochs):

        train(args, epoch+1, dataset, objective)
        with torch.no_grad():
            model.eval()
            score10, score5 = calculateRecall(dataset)
            model.train(True)

        print(f"> Validation Recall10: {score10} and Recall5: {score5}")

        if score10 > max_recall:
            max_recall = score10
            save_model(args, epoch, max_recall)

def create_model(args):
    print("> Create model.")

    with open(os.path.join(args.data_path, "dict&vectors.pkl"), "rb") as f:
        [word2idx, vectors] = pickle.load(f)

    global model

    model = models.RNNattELMo(window_size=args.max_length,
                              hidden_size=args.hidden_size,
                              drop_p=args.drop_p,
                              num_of_words=len(word2idx),
                              rec_len=rec_len,
                              rep_len=rep_len
                            )
    model = model.to(device)
    print(model)

    global optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)

    return word2idx, vectors

def save_model(args, epoch, max_recall):
    print(">> Saving Model...")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'max_recall': max_recall
    }, args.model_dump)

def load_model(ckptname):
    print("> Loading..")
    ckpt = torch.load(ckptname)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
    return ckpt['max_recall']

def testAll(args):
    word2idx, vectors = create_model(args)
    global idx2word
    idx2word = {b:a for a,b in word2idx.items()}
    print("> Loading trained model and Test")
    max_recall = load_model(args.model_dump)
    print(f"max_recall: {max_recall}")
    test_data = load_test_data(args, word2idx)
    with torch.no_grad():
        model.eval()
        do_predict(args, test_data, idx2word)

def do_predict(args, test_data, idx2word):
    write = []
    for cou, data in enumerate(test_data):
        input_data_rec, input_data_rep = zip(*data)

        input_data_rec = torch.tensor(input_data_rec, dtype=torch.long)
        input_data_rep = torch.tensor(input_data_rep, dtype=torch.long)

        input_data_rec = input_data_rec.to(device)
        input_data_rep = input_data_rep.to(device)

        pred = model(input_data_rec, input_data_rep, elmo)


        if args.attn == 1 or args.attn == 3:
            pred = pred[0]
        pred1 = nn.Sigmoid()(pred)
        out = pred.detach().cpu().numpy().argsort()[::-1].tolist()[:10]
        out1 = pred1.detach().cpu().numpy().argsort()[::-1].tolist()[:10]

        out = "".join(["1-" if i in out else "0-" for i in range(100)])
        write.append((cou+9000001, out))

    df = pd.DataFrame(write, columns=['Id', 'Predict'])
    df.to_csv(args.output_csv, index=None)

def main(args):
    # init
    # global loss_epoch_tr
    # global acc_epoch_tr
    # global loss_epoch_va
    # global acc_epoch_va

    # loss_epoch_tr = []
    # acc_epoch_tr = []
    # loss_epoch_va = []
    # acc_epoch_va = []

    if args.train:
        trainIter(args)
    testAll(args)

if __name__ == '__main__':
    print(device)
    print(rec_len, rep_len)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-hn', '--hidden_size', type=int, default=128)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-dr', '--drop_p', type=float, default=0.2)
    parser.add_argument('-md', '--model_dump', type=str, default='./model.tar')
    parser.add_argument('-ml', '--model_load', type=str, default=None, help='Print every p iterations')
    parser.add_argument('-o', '--output_csv', type=str, default='./output.csv')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('-s', '--save_iter', type=int, default=30, help='Save every p iterations')
    parser.add_argument('--max_length', type=int, default=128, help='Max dialogue length')
    parser.add_argument('-tr', '--train', type=int, default=1, help='Train and test: 1, Only test: 0')
    parser.add_argument('-atn', '--attn', type=int, default=1, help='Attn RNN: 1, RNN: 0')
    args = parser.parse_args()
    # print(args.model_load)
    main(args)