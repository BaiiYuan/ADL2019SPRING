import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from IPython import embed

def draw(tmp, model_name):
    print(model_name)
    print(np.mean(tmp))
    mean_batch = 100
    train_x, reward = zip(*tmp)
    train_x = list(map(int, train_x))
    print(np.mean(reward))
    reward = [np.mean(reward[i:i+mean_batch]) for i in range(len(reward))]
    plt.plot(train_x[:-mean_batch], reward[:-mean_batch], '-', label=model_name)



def draw_2(tmp, model_name):
    mean_batch = 100
    print(model_name)
    plt.figure(figsize=(20,10))
    train_x, reward = zip(*tmp)
    train_x = list(map(int, train_x))
    print(np.mean(reward))
    reward = [np.mean(reward[i:i+mean_batch]) for i in range(len(reward))]
    plt.plot(train_x[:-mean_batch], reward[:-mean_batch], '-', label=model_name)


    tmp2 = np.load("dqn_0.99.npy")
    train_x, reward = zip(*tmp2)
    print(len(train_x))
    train_x = list(map(int, train_x))
    print(np.mean(reward))
    reward = [np.mean(reward[i:i+mean_batch]) for i in range(len(reward))]
    plt.plot(train_x[:-mean_batch], reward[:-mean_batch], '-', label="dqn")

    plt.title("Learning Curves of DQN")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    plt.legend(loc=4)
    plt.savefig("./pic/{}.png".format(model_name))
    plt.clf()

def draw_4():
    mean_batch = 100
    plt.figure(figsize=(20,10))
    model_name = "dqn"
    tmp2 = np.load("dqn_0.99.npy")
    train_x, reward = zip(*tmp2)
    print(len(train_x))
    train_x = list(map(int, train_x))
    print(np.mean(reward))
    reward = [np.mean(reward[i:i+mean_batch]) for i in range(len(reward))]
    plt.plot(train_x[:-mean_batch], reward[:-mean_batch], '-', label="dqn")

    plt.title("Learning Curves of DQN")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    plt.legend(loc=4)
    plt.savefig("./pic/{}.png".format(model_name))
    plt.clf()

def draw_3(tmp, model_name):
    print(model_name)
    print(np.max(tmp))
    mean_batch = 100
    reward = [np.mean(tmp[i:i+mean_batch]) for i in range(len(tmp)-mean_batch)]
    train_x = [i for i in range(1, len(reward)+1)]

    plt.figure(figsize=(20,10))
    plt.plot(train_x, reward, '-', label=model_name)

    plt.title("Learning Curves of PG")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.legend()
    plt.savefig("./pic/{}.png".format(model_name))
    plt.clf()


draw_3(np.load("pg.npy"), "pg")

draw_4()

draw_2(np.load("duel_dqn_0.99.npy"), "duel dqn")
draw_2(np.load("double_dqn_0.99.npy"), "double dqn")
draw_2(np.load("noisy_dqn_0.99.npy"), "noisy dqn")
draw_2(np.load("prioritized_dqn_0.99.npy"), "prioritized dqn")



plt.figure(figsize=(20,10))
draw(np.load("dqn_0.99.npy"), "dqn with gamma = 0.99")
draw(np.load("dqn_0.95.npy"), "dqn with gamma = 0.95")
draw(np.load("dqn_0.9.npy"), "dqn with gamma = 0.9")
draw(np.load("dqn_0.85.npy"), "dqn with gamma = 0.85")
plt.title("Learning Curves of DQN")
plt.xlabel("Steps")
plt.ylabel("Reward")

plt.legend(loc=4)
plt.savefig("./pic/dqn_gamma_compare.png")
plt.clf()