import os
import pandas as pd
import numpy as np
from IPython import embed

path = "./csv"
filenames = ["bi2_gru_-3_att_3_VER2.csv",
             "bi2_gru_-3_att_3_VER3.csv",
             "bi2_gru_-3_att_3_VER4.csv",
             "bi2_gru_-3_att_3_VER5.csv"]

tmp_results = []
for filename in filenames:
    tmp = pd.read_csv(os.path.join(path, filename)).values[:, 1].tolist()
    tmp = np.array([i[:-1].split("-") for i in tmp]).astype(int)
    tmp_results.append(tmp)

results = np.array(tmp_results).sum(0)*3

filenames = ["bi2_gru_-3_B_3.csv",
             "bi2_gru_-3_att_3.csv",
             "cosine_gensim_300.csv",
             "cosine_gensim_1000.csv"]

tmp_results = []
for filename in filenames:
    tmp = pd.read_csv(os.path.join(path, filename)).values[:, 1].tolist()
    tmp = np.array([i[:-1].split("-") for i in tmp]).astype(int)
    tmp_results.append(tmp)

results += np.array(tmp_results).sum(0)

results = results.tolist()
write = [
]
for cou, item in enumerate(results):
    print(np.array(item))
    out = np.array(item).argsort()[::-1].tolist()[:10]
    out = "".join(["1-" if i in out else "0-" for i in range(100)])
    write.append((cou+9000001, out))

df = pd.DataFrame(write, columns=['Id', 'Predict'])
df.to_csv("ensemble2.csv", index=None)
