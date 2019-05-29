import argparse
import sys
import ipdb
import random

import torch
import torchvision.transforms as transforms
from IPython import embed

from trainer import GANtrainer
from data import Cartoonset100kDataset
from argument import add_arguments, USE_CUDA, device

# Set random seem for reproducibility
manualSeed = 1126 # random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if USE_CUDA:
    torch.cuda.manual_seed_all(manualSeed)

def main(args):
    dataset = Cartoonset100kDataset(attr_txt="./data/selected_cartoonset100k/cartoon_attr.txt",
                                    root_dir="./data/selected_cartoonset100k/images/",
                                    transform=transforms.Compose([
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
        parser = add_arguments(parser)
        args = parser.parse_args()
        main(args)
