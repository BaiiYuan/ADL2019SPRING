import argparse
import sys
import ipdb
from IPython import embed

from train import setting_seed
from trainer import GANtrainer
from argument import add_arguments, USE_CUDA, device

def main(args):
    trainer = GANtrainer(args)
    trainer.init_model()
    trainer.gen_output(iters=5)

if __name__ == '__main__':
     with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser = add_arguments(parser)
        args = parser.parse_args()
        main(args)