# ADL Homework 3 Report

> b05902002 資工三 李栢淵



### 1. Basic Performance (6%)

Describe your Policy Gradient & DQN model (1% + 1%)

Plot the learning curve to show the performance of your Policy Gradient on **LunarLander** (2%)

Plot the learning curve to show the performance of your DQN on **Assualt** (2%)

X-axis: number of time steps

Y-axis: average reward in last n episodes. You can arbitrarily choose n to make your figure clea

### 2. Experimenting with DQN hyperparameters (2%)

Choose one hyperparameter of your choice and run at least three other settings of this hyperparameter

You should find a hyperparameter that makes a nontrivial difference to DQN.
- For example, if you just choose network hidden size in {256, 1126, 10000, 20},  you might not get full score in this part.

Plot all four learning curves in the same figure (1%)

Explain why you choose this hyperparameter and how it affect the results (0.5% + 0.5%)

Candidates: gamma, network architecture, exploration schedule/rule, target network update frequency, etc.
You can use any environment to show your results

### 3. Improvements to Policy Gradient & DQN / Other RL methods (2% + 2%)

Choose two improvements to PG & DQN or other RL methods.

Other RL methods include
- Actor-Critic series (A2C, A3C, ACKTR etc.)
- DDPG, Curiosity-Driven Learning, AlphaStar etc.

For each method you choose,
- describe why they can improve the performance (1%)
- plot the graph to compare results with and without improvement (1%)

You can train on any environment to show your results, so you should better choose environment where you can see significant difference between those methods.

Grading will simultaneously consider your description and actual model performance.
