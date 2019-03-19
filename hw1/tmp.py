import torch
import re
import os
import json
import gensim
import pickle
from gensim.models import Word2Vec
from IPython import embed
import numpy as np
import pandas as pd

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "./data"

with open(os.path.join(data_path, "test.pkl"), "rb") as f:
	a = pickle.load(f)

tmp = " ".join([" ".join(i) for i in a[0]['records']])
print(len(tmp.split()))
wa = a[0]['wrong_answer']
print(np.mean([len(i) for i in wa]))

with open(os.path.join(data_path, "valid.pkl"), "rb") as f:
	a = pickle.load(f)

embed()