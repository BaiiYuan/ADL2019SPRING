#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python code/test.py -tl $1 -od $2
# bash cgan.sh ./data/sample_test/sample_fid_testing_labels.txt test_output
# bash code/cgan.sh ./data/sample_test/sample_fid_testing_labels.txt test_output
