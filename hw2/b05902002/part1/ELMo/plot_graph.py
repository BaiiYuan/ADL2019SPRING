import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

def plot(train, valid, name):
	valid_x = [(i+1)*1000 for i in range(93)]

	train = [train[i*30938:(i+1)*30938] for i in range(3)]
	train = [[np.mean(ep[i*100:(i+1)*100]) for i in range(310)] for ep in train]
	train = np.array(train).reshape(-1)
	train_x = [(i+1)*100 for i in range(930)]

	plt.figure(figsize=(20,10))
	plt.plot(train_x, train, '-', label='train')
	plt.plot(valid_x, valid, '-', label='valid')

	plt.title(f"{name} Loss")
	plt.xlabel("Iterations (batch size: 32)")
	plt.ylabel("Loss")

	plt.legend()
	plt.savefig(f"{name}.png")


train = np.load("train_loss.npy")
valid = np.load("valid_loss.npy")

train_f, train_b = train[:, 0], train[:, 1]
valid_f, valid_b = valid[:, 0], valid[:, 1]


plot(train_f, valid_f, name="Forward")
plot(train_b, valid_b, name="Backward")
plot((train_f+train_b)/2, (valid_f+valid_b)/2, name="Mean")
