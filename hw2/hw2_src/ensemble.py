import os
import pandas as pd
import numpy as np
from IPython import embed

path = "./"

f1 = "bert3"

out = [pd.read_csv(os.path.join(f1, f"epoch-17.csv")) for i in range(15, 20)]
out = [ np.eye(6)[df.values[:,1]] for df in out]
out = np.array(out).sum(axis=0).argmax(axis=1).tolist()
out = [(cou+20001, item) for cou, item in enumerate(out)]


df = pd.DataFrame(out, columns=['Id', 'label'])
df.to_csv("ensemble_bert3.csv", index=None)
