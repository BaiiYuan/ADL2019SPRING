import os
import sys
import pandas as pd
import numpy as np

path = "./"

filenames = [f"RNN_attn_self_ver{i}.csv" for i in [0,3,4,6,7,8,9]]# range(11)]

tmp_results = []
for filename in filenames:
    tmp = pd.read_csv(os.path.join(path, filename)).values[:, 1].tolist()
    tmp = np.array([i[:-1].split("-") for i in tmp]).astype(int)
    tmp_results.append(tmp)

results = np.array(tmp_results).sum(0)

results = results.tolist()
write = []

for cou, item in enumerate(results):
    # print(np.array(item))
    out = np.array(item).argsort()[::-1].tolist()[:10]
    out = "".join(["1-" if i in out else "0-" for i in range(100)])
    write.append((cou+9000001, out))

df = pd.DataFrame(write, columns=['Id', 'Predict'])
df.to_csv(sys.argv[1], index=None)
