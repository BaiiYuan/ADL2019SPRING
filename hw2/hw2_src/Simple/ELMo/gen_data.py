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

t = time()
print("Loading word2idx... ", end="")
with open(os.path.join(data_path, "word2idx-vectors.pkl"), "rb") as f:
    word2idx, vectors = pickle.load(f)
print(f"Consuming Time: {time()-t}");
print(len(word2idx.keys()), vectors.shape)

def preprocessSentence(raw_sentence):
    a = raw_sentence
    a = a.lower()
    a = re.sub(r"[^a-zA-Z!?',.]+", r" ", a)
    a = nltk.word_tokenize(a)
    a = [word2idx.get(i, 0) for i in a]
    return a

def main():
    t1 = time()
    print("Loading... ", end="")
    with open(os.path.join(data_path, "language_model/corpus.txt")) as f:
        a = f.read().strip().split("\n")
    print(f"Consuming Time: {time()-t1}"); t1 = time()

    print("Construct Pool...", end="")
    pool = mp.Pool(processes=64)
    print(f"Consuming Time: {time()-t1}"); t1 = time()
    print("Go Pool!", end="")
    res = pool.map(preprocessSentence, a)
    print(f"Consuming Time: {time()-t1}"); t1 = time()

    print("Saving... ")
    with open(os.path.join(data_path, "all_sentence.pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"Consuming Time: {time()-t1}"); t1 = time()

if __name__ == '__main__':
    main()