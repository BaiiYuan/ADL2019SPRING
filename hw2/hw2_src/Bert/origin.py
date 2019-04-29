import torch
import torch.nn as nn
import torch.optim as optim

import os
import re
import sys
import time
import ipdb
import random
import argparse
import numpy as np
import pandas as pd

from sys import stdout
from IPython import embed
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
BERT = 'bert-large-cased'

def _pad_and_cut(seq, max_leng, pad, array=True):
    if len(seq) < max_leng:
        seq = seq + [pad]*(max_leng-len(seq))
    else:
        seq = seq[:max_leng]
    if array:
        return np.array(seq)
    return seq

def process_sent(s):
    # s = re.sub(r"[^a-zA-Z!?',.]+", r" ", s)
    # s = re.sub(r"([!?])", r" \1 ", s)
    # s = re.sub(r"\s+", r" ", s).strip()
    return s

def process_df(df, tokenizer):
    sents = df.values[:, 1].tolist()
    labels = df.values[:, 2].tolist()
    num = df.shape[0]

    sents = [tokenizer.tokenize(sent) for sent in sents]
    sents = [tokenizer.convert_tokens_to_ids(sent) for sent in sents]
    sents = [_pad_and_cut(sent, args.max_length, 0) for sent in sents]

    return [(sents[i], int(labels[i]-1)) for i in range(num)]


def load_data(args):
    print("[*] Loading data...")
    dataset = {}
    tokenizer = BertTokenizer.from_pretrained(BERT, do_lower_case=False, do_basic_tokenize=True)
    df_train = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    df_dev = pd.read_csv(os.path.join(args.data_path, "dev.csv"))
    df_test = pd.read_csv(os.path.join(args.data_path, "test.csv"))

    train_data = process_df(df_train, tokenizer)
    valid_data = process_df(df_dev, tokenizer)

    dataset["train"], dataset["dev"] = train_data, valid_data # train_test_split(train_data+valid_data, test_size=0.1)
    print(len(dataset["train"]), len(dataset["dev"]))
    dataset["test"] = process_df(df_test, tokenizer)

    return dataset, tokenizer

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

        input_data = torch.tensor(input_data, dtype=torch.long)
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

        input_data = torch.tensor(input_data, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        yield input_data.to(device), labels.to(device)

def train(args, epoch, dataset, criterion, max_acc):
    print("\n------------------------------------------")
    gen = data_generator(args, dataset["train"], args.batch_size, shuffle=True)  # generate train data

    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    model.train(True)

    for idx, (input_data, labels) in enumerate(gen):
        # Forward and backward.
        optimizer.zero_grad()
        pred = model(input_data)

        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        loss = loss.data.cpu().item()
        epoch_loss.append(loss)

        acc = (pred.argmax(dim=1) == labels).float().cpu().tolist()
        epoch_acc.extend(acc)

        Iter = 100.0 * (idx + 1) / steps_per_epoch
        stdout.write("\r> Train: IDx: {}. Epoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}%".format(idx,epoch,
                                                                                                       args.epochs,
                                                                                                       Iter,
                                                                                                       np.mean(epoch_loss),
                                                                                                       np.mean(epoch_acc)*100
                                                                                                       ))
        if (idx + 1) % args.print_iter == 0 :
            print(" ")
            with torch.no_grad():
                model.eval()
                tmp_acc = valid(args, dataset, criterion)
            if tmp_acc > max_acc:
                max_acc = tmp_acc
                save_model(args, epoch, max_acc)
            model.train(True)
            print("------------------------------------------")


    print("\n> Spends {:.2f} seconds.".format(time.time() - t1))
    print("[*] The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100,
                                                                                np.mean(epoch_loss)))
    return max_acc

def valid(args, dataset, criterion):
    print("------------------------------------------")
    gen = valid_data_generator(args, dataset["dev"], args.batch_size, shuffle=False)  # generate train data
    t1 = time.time()
    epoch_loss = []
    epoch_acc = []
    for idx, (input_data, labels) in enumerate(gen):
        pred = model(input_data)
        loss = criterion(pred, labels)

        loss = loss.data.cpu().item()
        epoch_loss.append(loss)

        acc = (pred.argmax(dim=1) == labels).float().cpu().tolist()
        epoch_acc.extend(acc)

        stdout.write("\r> Valid: Idx: {}, Loss: {:.4f}, Acc: {:.2f}%".format(idx,
                                                                             np.mean(epoch_loss),
                                                                             np.mean(epoch_acc)*100,
                                                                             ))

    print("\n> Spends {:.2f} seconds.".format(time.time() - t1))
    print("[-] The Validation dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100,
                                                                                  np.mean(epoch_loss)))
    return np.mean(epoch_acc)

def trainIter(args):
    dataset, tokenizer, criterion, max_acc = trainInit(args)
    print(max_acc)
    for epoch in range(args.epochs):
        max_acc = train(args, epoch+1, dataset, criterion, max_acc)
        with torch.no_grad():
            model.eval()
            tmp_acc = valid(args, dataset, criterion)
        if tmp_acc > max_acc:
            max_acc = tmp_acc
            save_model(args, epoch, max_acc)


def trainInit(args):
    max_acc = 0.
    dataset, tokenizer = load_data(args)
    create_model(args, dataset)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = criterion.to(device)

    return dataset, tokenizer, criterion, max_acc

def create_model(args, dataset):
    print("[*] Create model.")

    global model
    model = BertForSequenceClassification.from_pretrained(BERT, num_labels=5)
    # for i in model.bert.named_parameters():
    #     i[1].requires_grad=False

    model = model.to(device)
    # print(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(
            len(dataset["train"]) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    global optimizer

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr_rate,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)

    # optimizer = optim.Adam(model.parameters(),
    #                        lr=args.lr_rate) # , betas=(0.9, 0.999), weight_decay=1e-3)
    return

def save_model(args, epoch, max_acc):
    print("[*] Saving Model")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'max_acc': max_acc
    }, args.model_dump)

def load_model(ckptname, train=False):
    print("> Loading..")
    ckpt = torch.load(ckptname)
    model.load_state_dict(ckpt['model'])
    if train:
        optimizer.load_state_dict(ckpt['opt'])
    return ckpt['max_acc']

def testAll(args):
    dataset, tokenizer = load_data(args)
    create_model(args, dataset)
    print("> Loading trained model and Test")
    max_acc = load_model(args.model_dump)
    print(f"max_acc: {max_acc}")

    # criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    # print(valid(args, dataset, criterion))

    with torch.no_grad():
        model.eval()
        do_predict(args, dataset["test"])

def do_predict(args, test_data):
    write = []
    gen = valid_data_generator(args, test_data, args.batch_size, shuffle=False)
    for idx, (input_data, labels) in enumerate(gen):
        pred = model(input_data)

        out = pred.argmax(dim=1).detach().cpu().numpy().tolist()
        write.extend(out)

    print(len(write))
    write = [(c+20001, o+1) for c,o in enumerate(write)]

    df = pd.DataFrame(write, columns=['Id', 'label'])
    df.to_csv(args.output_csv, index=None)

def main(args):
    if args.train:
        trainIter(args)
    testAll(args)

if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser.add_argument('-dp', '--data_path', type=str, default='../data')
        parser.add_argument('-e', '--epochs', type=int, default=10)
        parser.add_argument('-b', '--batch_size', type=int, default=16)
        parser.add_argument('-hn', '--hidden_size', type=int, default=512)
        parser.add_argument('-lr', '--lr_rate', type=float, default=1e-5)
        parser.add_argument('-dr', '--drop_p', type=float, default=0.5)
        parser.add_argument('-md', '--model_dump', type=str, default='./bert_model_ver3.tar')
        parser.add_argument('-ml', '--model_load', type=str, default=None, help='Model Load')
        parser.add_argument('-p', '--print_iter', type=int, default=271, help='Print every p iterations')
        parser.add_argument('-mc', '--max_count', type=int, default=40)
        parser.add_argument('-tr', '--train', type=int, default=1)
        parser.add_argument('-as', '--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument('-o', '--output_csv', type=str, default="out.csv")
        parser.add_argument('-l', '--max_length', type=int, default=64, help='Max sequence length')
        args = parser.parse_args()
        main(args)