# ADL HW3 Training

> B05902002 資工三 李栢淵



### Training

##### Policy Gradient

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --train_pg
```

##### DQN

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --train_dqn
```

##### Plotting Figure

After training there are record's npy files containing the training 

I put the other code in `b05902002/improvement`

```
Flags:
	DOUBLE_DQN
	DUEL_DQN
	NOISY_DQN
	PRIORITIZED_DQN
if the Flag is 1, it will train the certain improvement version of DQN.
```

Training:

```sh
cd improvement
CUDA_VISIBLE_DEVICES=0 python main.py --train_dqn --gamma 0.99 # with different flag
# and 
CUDA_VISIBLE_DEVICES=0 python main.py --train_dqn --gamma 0.95
CUDA_VISIBLE_DEVICES=0 python main.py --train_dqn --gamma 0.9
CUDA_VISIBLE_DEVICES=0 python main.py --train_dqn --gamma 0.85
```



After training:

```sh
python draw.py
```

