#!/bin/bash
for (( i = $2; i < $3; i++ ))
do
    echo CUDA_VISIBLE_DEVICES=$1 python main.py -md preserve/BiDAF_200_200_128_D2_b_ver${i}.tar -lr 1e-3 -dr 0.2 -hn 128 -tr 1 -e 15 -dp ./Uncase -b 100 -atn 2 -o BiDAF_200_200_128_D2_b_ver${i}.csv

    CUDA_VISIBLE_DEVICES=$1 python main.py -md preserve/BiDAF_200_200_128_D2_b_ver${i}.tar -lr 1e-3 -dr 0.2 -hn 128 -tr 1 -e 15 -dp ./Uncase -b 100 -atn 2 -o BiDAF_200_200_128_D2_b_ver${i}.csv
done
