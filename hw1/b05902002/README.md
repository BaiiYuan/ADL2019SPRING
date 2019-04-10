# ADL HW1 Training

> B05902002 資工三 李栢淵



### Preprocessing

```sh
cd train
mkdir data_dir
mv /PATH/train.json /PATH/test.json /PATH/valid.json ./data_dir
mv /PATH/crawl-300d-2M.vec .
python preprocess.py ./data_dir 
```



### Training and Testing 

##### run w/o attention

```sh
CUDA_VISIBLE_DEVICES=0 python main.py -md RNN_attn.tar -lr 1e-3 -tr 1 -e 30 -dp ./data_dir -atn 3 -dr 0.4 -hn 128 -o output_attn.csv
```

##### rnn w/ attention

```sh
CUDA_VISIBLE_DEVICES=0 python main.py -md RNN_base.tar -lr 1e-3 -tr 1 -e 30 -dp ./data_dir -atn 0 -dr 0.2 -hn 256 -o output_base.csv
```

if you need test, just assure that the model exist and change the argument `-tr 0`

##### Ensemble Training script

```sh
#!/bin/bash
mkdir preserve
for (( i = 0; i <= 10; i++ ))
do
    CUDA_VISIBLE_DEVICES=0 python main.py -md preserve/RNN_attn_self_ver${i}.tar -lr 1e-3 -tr 1 -e 20 -dp ./data_dir -atn 3 -dr 0.4 -hn 128 -o RNN_attn_self_ver${i}.csv
done
python ensemble.py output.csv
```



### Draw Attention

```sh
mkdir atten_pic
python plot_attention.py -md RNN_attn.tar -dp ./data_dir -atn 3
```

