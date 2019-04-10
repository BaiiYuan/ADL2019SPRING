import torch
import re
import os
import sys
import json
import pickle
from sys import stdout
import nltk

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "./hw1_upload"


def preprocessSentence(raw_sentence):
    a = raw_sentence
    a = re.sub(r"[^a-zA-Z!?',.]+", r" ", a)
    a = a.lower()
    a = nltk.word_tokenize(a)
    a.append("<pad>")
    return a


def preprocessTestData(raw_datas):
    ret_data = []
    cou = 0
    for data in raw_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        tmp = {}
        records = data['messages-so-far']
        options = data['options-for-next']
        thisID = data['example-id']

        wrong_answer = []
        for i, option in enumerate(options):
            wrong_answer.append(preprocessSentence(option['utterance']))

        tmp['thisID'] = thisID
        tmp['records'] = [preprocessSentence(record['utterance']) for record in records]
        tmp['wrong_answer'] = wrong_answer

        ret_data.append(tmp)


        assert(len(tmp['wrong_answer'])==100)
    print(" ")
    return ret_data

def morePreprocessTestData(processed_datas, word2idx):
    ret_data = []
    cou = 0
    for data in processed_datas:
        stdout.write("\r{}".format(cou))
        cou+=1
        records = data['records']
        data['records'] = [ [word2idx.get(i, 1) for i in record] for record in records]
        wrong_answer = data['wrong_answer']
        data['wrong_answer'] = [ [word2idx.get(i, 1) for i in record] for record in wrong_answer]
        ret_data.append(data)
    print(" ")
    return ret_data

def main():
    print("loading data ...")

    with open(sys.argv[1]) as f:
        test_raw = json.load(f)

    print("processing data ...")
    test_data = preprocessTestData(test_raw)

    with open(os.path.join(data_path, "dict&vectors.pkl"), "rb") as f:
        [word2idx, vectors] = pickle.load(f)

    test_data = morePreprocessTestData(test_data, word2idx)

    with open(os.path.join(data_path, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    main()