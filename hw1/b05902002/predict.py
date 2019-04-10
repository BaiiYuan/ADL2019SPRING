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
from tqdm import tqdm

from load import load_data, load_test_data
from config import rep_len, rec_len, RATE

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(args):
    print("> Create model.")

    ## Gensim
    # word_model = Word2Vec.load("Word2Vec_V1.h5")
    # vectors = word_model.wv
    # all_words = vectors.index2word
    # mean_vector = vectors.vectors.mean(axis=0)
    # wei = torch.tensor(vectors.vectors, dtype=torch.float)
    ## Gensim

    with open(os.path.join(args.data_path, "dict&vectors.pkl"), "rb") as f:
        [word2idx, vectors] = pickle.load(f)

    global model
    if args.attn == 3:
        model = models.RNNatt(window_size=args.max_length,
                              hidden_size=args.hidden_size,
                              drop_p=args.drop_p,
                              num_of_words=len(word2idx),
                              rec_len=rec_len,
                              rep_len=rep_len
                            )
    else: # args.attn == 0
        model = models.RNNbase(window_size=args.max_length,
                               hidden_size=args.hidden_size,
                               drop_p=args.drop_p,
                               num_of_words=len(word2idx)
                            )


    model.word_embedding.load_state_dict({'weight': vectors.to(torch.float32)})
    model.word_embedding.weight.requires_grad = False

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
    if device == "cpu":
        ckpt = torch.load(ckptname, map_location='cpu')
    else:
        ckpt = torch.load(ckptname)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
    return ckpt['max_recall']

def testAll(args):
    word2idx, vectors = create_model(args)
    print("> Loading trained model and Test")
    max_recall = load_model(args.model_dump)
    print(f"max_recall: {max_recall}")
    test_data = load_test_data(args, word2idx)
    with torch.no_grad():
        model.eval()
        do_predict(args, test_data)

def do_predict(args, test_data):
    write = []
    for cou, data in enumerate(test_data):
        input_data_rec, input_data_rep = zip(*data)

        input_data_rec = torch.tensor(input_data_rec, dtype=torch.long)
        input_data_rep = torch.tensor(input_data_rep, dtype=torch.long)

        input_data_rec = input_data_rec.to(device)
        input_data_rep = input_data_rep.to(device)

        pred = model(input_data_rec, input_data_rep)
        if args.attn == 3:
            pred = pred[0]
        pred1 = nn.Sigmoid()(pred)
        out = pred.detach().cpu().numpy().argsort()[::-1].tolist()[:10]
        out1 = pred1.detach().cpu().numpy().argsort()[::-1].tolist()[:10]

        out = "".join(["1-" if i in out else "0-" for i in range(100)])
        write.append((cou+9000001, out))

    df = pd.DataFrame(write, columns=['Id', 'Predict'])
    df.to_csv(args.output_csv, index=None)

def main(args):

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