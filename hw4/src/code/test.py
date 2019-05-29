import argparse
import sys
import ipdb
from IPython import embed

from trainer import GANtrainer
from argument import add_arguments, USE_CUDA, device

label_txt = ["./data/sample_test/sample_fid_testing_labels.txt",
             "./data/sample_test/sample_human_testing_labels.txt"]

def main(args):
    trainer = GANtrainer(args)
    trainer.init_model()
    trainer.gen_output(filename=args.testing_labels)

if __name__ == '__main__':
     with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser = add_arguments(parser)
        args = parser.parse_args()
        main(args)