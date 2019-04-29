#!/bin/bash
python3.6 main.py -tp $1 -lr 5e-6 -p 2220 -b 8 -tr 0 -e 10 -md bert_model_ver5.tar -l 64 -o $2 -bert bert-large-uncased