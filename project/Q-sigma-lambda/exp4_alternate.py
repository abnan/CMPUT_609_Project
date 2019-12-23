#Exponential decay

import numpy as np
import random
from windy_gridworld import WindyGridworldEnv, StochasticWindyGridworldEnv
import itertools
import matplotlib.pyplot as plt
import csv


np.random.seed(98)
env = StochasticWindyGridworldEnv()

episodes = 100
num_run = 10
# epsilon = 0.1
lr = 0.6
gamma = 1
lmbda = 0.7

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

def getSelectionProb(state, epsilon):
    maxProbAction = np.argmax(Q[state])
    selectionProb = np.repeat(epsilon/len(Q[state]), len(Q[state]))
    selectionProb[maxProbAction] += 1 - epsilon
    return selectionProb

def getEpsilonGreedyAction(state, epsilon):
    selectionProb = getSelectionProb(state, epsilon)
    finalSelectedAction = np.random.choice(np.arange(len(Q[state])), p = selectionProb)
    return finalSelectedAction

# f = open("results_exp1.csv", "w")
colours = ['r-','b-','g-','c-','y-','k-']
colour_index = 0
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
write_data = ','.join(str(x) for x in alphas)
#write header
# f.write("sigma,"+write_data+"\n")
avg_reward_list = list()
stats_episode_length = list()
stats_episode_reward = list()
returns_dict = dict()
for run in range(num_run):
    Q=np.zeros((env.observation_space.n, env.action_space.n))
    sigma=1
    sigma_avg=1
    for epi_num in range(episodes):
        z=np.zeros((env.observation_space.n, env.action_space.n))
        print(lr, run, epi_num)
        s = env.reset()
        a = getEpsilonGreedyAction(s, epsilon=0.1)
        episode_reward = 0
        td_error_list_this_episode = list()
        for epi_len in itertools.count():
            # env.render()
            s_n, reward, done, _ = env.step(a)
            episode_reward += reward
            a_n = getEpsilonGreedyAction(s_n, epsilon=0.1)
            # sigma = (1/np.e)**(epi_num)
            # sigma = 1/(epi_num+1)**2
            # print(sigma)
            td_target = reward + gamma * (sigma * Q[s_n, a_n] + (1-sigma) * np.dot(Q[s_n], getSelectionProb(s_n, epsilon=0)))
            td_error = td_target - Q[s, a]
            if(epi_num>0):
                sigma=clamp(abs(td_error/td_max), 0, 1)
            td_error_list_this_episode.append(abs(td_error))
            z[s, a] += 1
            Q += lr * td_error * z
            z = gamma * lmbda  * z * (sigma + (1 - sigma) * getSelectionProb(s_n, epsilon=0)[a_n])
            s = s_n
            a = a_n
            if done:
                if(epi_num in returns_dict):
                    returns_dict[epi_num].append(episode_reward)
                else:
                    returns_dict[epi_num] = [episode_reward]
                td_max = np.max(td_error_list_this_episode)
                # print(str(epi_num) + " ended in " + str(epi_len) + " steps. Total reward: " + str(episode_reward))
                # stats_episode_length.append(epi_len)
                #total episode return after that episode
                # stats_episode_reward.append(episode_reward)
                # sigma=sigma*0.95
                break

avg_rewards = []
for i in range(10, episodes):
    avg_rewards.append(np.mean(returns_dict[i]))
# rc_moving_avg_rewards = []
# for i in range(10, len(avg_rewards)):
#     rc_moving_avg_rewards.append(np.mean(avg_rewards[i-30:i]))
plt.plot(np.arange(len(avg_rewards)), avg_rewards, label = "Dynamic decay sigma 0.99; lambda=0.7")
plt.legend()
plt.show()
print(avg_rewards)
