import torch
import re
import os
import pickle
from IPython import embed
from sys import stdout
import nltk

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
data_path = "../data"



def main():
	pass

if __name__ == '__main__':
	main()