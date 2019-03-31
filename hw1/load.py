import os
import random
import pickle
import numpy as np
from IPython import embed
from sys import stdout

rep_len = 200
rec_len = 200
RATE = 4

def cut_to_length(max_length, arr, word2idx, rec=True):
    if len(arr) > max_length:
        if rec:
            arr = arr[-max_length:]
        else:
            arr = arr[:max_length]
    if len(arr) < max_length:
        arr = [0 for _ in range(max_length-len(arr))] + arr
    assert(len(arr)==max_length)
    return arr

def flatten2D(List2D):
     return [item for sublist in List2D for item in sublist]

def random_sample(dataset, args, word2idx, vectors, rate=RATE): # 1e5
    for cou in range(len(dataset)):
        records = dataset[cou]['records']
        dataset[cou]['records'] = flatten2D(records)

    vectors = vectors.numpy()
    out = []

    for cou, i in enumerate(dataset):
        stdout.write(f"\r{cou}")
        wa = i['wrong_answer']
        ca = i['correct_answer']
        records = i['records']

        ca_mean = np.array([vectors[i] for i in ca]).mean(axis=0)
        record_mean = np.array([vectors[i] for i in records]).mean(axis=0)

        ca = cut_to_length(rep_len, ca, word2idx, rec=False)
        records = cut_to_length(rec_len, records, word2idx)

        out.append([records, ca, 1])

        wa_sam = random.sample(wa, rate)

        for item in wa_sam:
            item = cut_to_length(rep_len, item, word2idx, rec=False)
            out.append([records, item, 0])

    return out

def process_valid_data(data_dict, word2idx):
    out = []
    records = data_dict['records']
    records = [item for sublist in records for item in sublist]
    records = cut_to_length(rec_len, records, word2idx)

    ca = data_dict['correct_answer']
    ca = cut_to_length(rep_len, ca, word2idx, rec=False)
    out.append([records, ca])

    wa = data_dict['wrong_answer']
    for item in wa:
        item = cut_to_length(rep_len, item, word2idx, rec=False)
        out.append([records, item])
    return out

def process_test_data(data_dict, word2idx):
    out = []
    records = data_dict['records']
    records = [item for sublist in records for item in sublist]
    records = cut_to_length(rec_len, records, word2idx)

    wa = data_dict['wrong_answer']
    for item in wa:
        item = cut_to_length(rep_len, item, word2idx)
        out.append([records, item])
    return out

def load_data(args, word2idx, vectors):
    print("> Load prepro-data...")
    dataset = {}

    with open(os.path.join(args.data_path, "valid.pkl"), "rb") as f:
        valid_data = pickle.load(f)

    tmp = []
    for item in valid_data:
        valid_out = process_valid_data(item, word2idx)
        tmp.append(valid_out)
    dataset['valid'] = tmp
    print("> Validation data finish")

    with open(os.path.join(args.data_path, "train.pkl"), "rb") as f:
        train_data = pickle.load(f)

    dataset['train'] = random_sample(train_data, args, word2idx, vectors)

    print("> Training data finish")
    print(len(dataset['train']))
    return dataset

def load_test_data(args, word2idx):
    print("> Load prepro-data...")

    with open(os.path.join(args.data_path, "test.pkl"), "rb") as f:
        valid_data = pickle.load(f)
    test_data = []
    for item in valid_data:
        test_data.append(process_test_data(item, word2idx))
    print("> Testing data finish")

    return test_data