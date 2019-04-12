import torch
import re
import os
import pickle
from IPython import embed
from sys import stdout
import nltk
import multiprocessing as mp
from time import time

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "../data"

with open(os.path.join(data_path, "word2idx-vectors-v2.pkl"), "rb") as f:
    [word2idx_new, vectors_small] = pickle.load(f)

def preprocessSentence(raw_sentence):
    a = raw_sentence.split()
    return a

def turn_to_idx(raw_sentence):
    a = raw_sentence.split()
    return [word2idx_new.get(word, 1) for word in a]


def load_vectors(f):
    cou = 3 # <pad> as 0
    word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2}
    vectors = []
    for line in f:
        tokens = line.rstrip().split(' ')
        if cou == 138703:
            word2idx["Baiyuan"] = cou
        else:
            word2idx[tokens[0]] = cou
        cou += 1
        stdout.write("\r{} {}".format(cou, len(word2idx)))
        vectors.append(list(map(float, tokens[1:])))
    print(len(word2idx.keys()), len(vectors))
    print(" ")
    vectors = torch.tensor(vectors)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.nn.init.uniform_(torch.empty(1, 300)), vectors], dim=0)
    vectors = torch.cat([torch.zeros(1, 300), vectors], dim=0)

    return word2idx, vectors

def main():
    # with open(os.path.join(data_path, "GloVe/glove.840B.300d.txt")) as f:
    #     word2dictid, vectors = load_vectors(f)
    # with open(os.path.join(data_path, "word2idx-vectors.pkl"), "wb") as f:
    #     pickle.dump([word2dictid, vectors], f)
    # del word2dictid, vectors

    t1 = time()

    # print("Loading word2idx-vectors... ", end="")
    # with open(os.path.join(data_path, "word2idx-vectors.pkl"), "rb") as f:
    #     [word2dictid, vectors] = pickle.load(f)
    # print(f"Consuming Time: {time()-t1}"); t1 = time()

    print("Loading corpus... ", end="")
    with open(os.path.join(data_path, "language_model/corpus_tokenized.txt")) as f:
        a = f.read().strip().split("\n")
    print(f"Consuming Time: {time()-t1}"); t1 = time()

    print("Construct Pool...", end="")
    pool = mp.Pool(processes=64)
    print(f"Consuming Time: {time()-t1}"); t1 = time()

    # print("Go Pool!", end="")
    # res = pool.map(preprocessSentence, a)
    # print(f"Consuming Time: {time()-t1}"); t1 = time()

    # print("Generate Wordset!", end="\n")
    # wordset = set(['<pad>', '<unk>', '<bos>'])
    # for cou, sent in enumerate(res):
    #     stdout.write("\r{}".format(cou))
    #     wordset.update(sent)
    # print(f"\nConsuming Time: {time()-t1}"); t1 = time()


    # print("Create dict again!", end="")
    # word2idx_new = {}
    # vectors_small = []
    # print(len(wordset))
    # for word in wordset:
    #     if word in word2dictid.keys():
    #         word2idx_new[word] = len(word2idx_new)
    #         vectors_small.append(vectors[word2dictid[word]].numpy())
    # vectors_small = torch.tensor(vectors_small)
    # print(len(word2idx_new))
    # print(vectors_small.shape)
    # print(f"Consuming Time: {time()-t1}"); t1 = time()
    # with open(os.path.join(data_path, "word2idx-vectors-v2.pkl"), "wb") as f:
    #     pickle.dump([word2idx_new, vectors_small], f)



    print("Go Pool!", end="")
    id_res = pool.map(turn_to_idx, a)
    print(f"Consuming Time: {time()-t1}"); t1 = time()

    print("Saving... ")
    with open(os.path.join(data_path, "all_sentence.pkl"), "wb") as f:
        pickle.dump(id_res, f)
    print(f"Consuming Time: {time()-t1}"); t1 = time()
    embed()

if __name__ == '__main__':
        main()