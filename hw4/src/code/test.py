import argparse
import sys
import ipdb
import random

import torch
import torchvision.transforms as transforms

from IPython import embed

from trainer import GANtrainer

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def main(args):

    trainer = GANtrainer(args)
    trainer.init_model()
    trainer.gen_output(iters=40)


if __name__ == '__main__':
     with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        main(args)