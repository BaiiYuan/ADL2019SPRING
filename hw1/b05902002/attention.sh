#!/bin/bash
python3.6 preprocess_test.py $1
python3.6 predict.py -md ./hw1_upload/RNN_attn_self_ver0.tar -lr 1e-3 -tr 0 -e 30 -dp ./hw1_upload -atn 3 -dr 0.4 -hn 128 -o $2