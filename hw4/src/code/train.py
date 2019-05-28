import argparse
import sys
import ipdb
import random

import torch
import torchvision.transforms as transforms

from IPython import embed

from trainer import GANtrainer
from data import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Set random seem for reproducibility
manualSeed = 999 # random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if USE_CUDA:
    torch.cuda.manual_seed_all(manualSeed)


def main(args):
    dataset = Cartoonset100kDataset(attr_txt="./data/selected_cartoonset100k/cartoon_attr.txt",
                                    root_dir="./data/selected_cartoonset100k/images/",
                                    transform=transforms.Compose([
                                              # transforms.Resize(64),
                                              # transforms.CenterCrop(64),
                                              # transforms.Rescale(256),
                                              # transforms.RandomCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    trainer = GANtrainer(args)
    trainer.init_dataset(dataset)
    trainer.init_model()
    # trainer.plot_test()
    trainer.train()


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser.add_argument('-dp', '--data_path', type=str, default='../data')
        parser.add_argument('-tp', '--test_path', type=str, default='../data/test.csv')
        parser.add_argument('-e', '--epochs', type=int, default=10)
        parser.add_argument('-b', '--batch_size', type=int, default=16)
        parser.add_argument('-hn', '--hidden_size', type=int, default=512)
        parser.add_argument('-lr', '--lr_rate', type=float, default=1e-5)
        parser.add_argument('-dr', '--drop_p', type=float, default=0.5)
        parser.add_argument('-md', '--model_dump', type=str, default='./bert_model_ver3.tar')
        parser.add_argument('-ml', '--model_load', type=str, default=None, help='Model Load')
        parser.add_argument('-p', '--print_iter', type=int, default=271, help='Print every p iterations')
        parser.add_argument('-mc', '--max_count', type=int, default=40)
        parser.add_argument('-tr', '--train', type=int, default=1)
        parser.add_argument('-o', '--output_csv', type=str, default="out.csv")
        parser.add_argument('-l', '--max_length', type=int, default=64, help='Max sequence length')
        parser.add_argument('-bert', '--bert_type', type=str, default='bert-large-uncased', help='Select Bert Type')
        args = parser.parse_args()
        main(args)
