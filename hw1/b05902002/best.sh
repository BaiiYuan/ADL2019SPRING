#!/bin/bash
python3.6 preprocess_test.py $1
# for (( i = 0; i <= 10; i++ ))
for i in 0 3 4 6 7 8 9;
do
    python3.6 predict.py -md hw1_upload/RNN_attn_self_ver${i}.tar -lr 1e-3 -tr 0 -e 20 -dp ./hw1_upload -atn 3 -dr 0.4 -hn 128 -o RNN_attn_self_ver${i}.csv
done

python3.6 ensemble.py $2