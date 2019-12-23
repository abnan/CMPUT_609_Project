"""
log decay
f(0) = 1

f(1) = log_2(1+0.99)

f(2) = log_2(1+log_2(1+0.99)) = log_2(1+f(1))
"""

import numpy as np
import random
from windy_gridworld import WindyGridworldEnv, StochasticWindyGridworldEnv
import itertools
import matplotlib.pyplot as plt
import csv
import math


np.random.seed(98)
env = StochasticWindyGridworldEnv()

episodes = 100
num_run = 100
# epsilon = 0.1
# lr = 0.5
gamma = 1
lmbda = 0.7


def getSelectionProb(state, epsilon):
    maxProbAction = np.argmax(Q[state])
    selectionProb = np.repeat(epsilon/len(Q[state]), len(Q[state]))
    selectionProb[maxProbAction] += 1 - epsilon
    return selectionProb

def getEpsilonGreedyAction(state, epsilon):
    selectionProb = getSelectionProb(state, epsilon)
    finalSelectedAction = np.random.choice(np.arange(len(Q[state])), p = selectionProb)
    return finalSelectedAction

f = open("results_exp6.csv", "w")
colours = ['r-','b-','g-','c-','y-','k-']
colour_index = 0
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
write_data = ','.join(str(x) for x in alphas)
#write header
f.write("sigma,"+write_data+"\n")
avg_reward_list = list()
for lr in alphas:
    stats_episode_length = list()
    stats_episode_reward = list()
    for run in range(num_run):
        Q=np.zeros((env.observation_space.n, env.action_space.n))
        for epi_num in range(episodes):
            z=np.zeros((env.observation_space.n, env.action_space.n))
            print(lr, run, epi_num)
            s = env.reset()
            a = getEpsilonGreedyAction(s, epsilon=0.1)
            episode_reward = 0
            sigma = 0.99
            for epi_len in itertools.count():
                # env.render()
                print(sigma)
                s_n, reward, done, _ = env.step(a)
                episode_reward += reward
                a_n = getEpsilonGreedyAction(s_n, epsilon=0.1)
                td_target = reward + gamma * (sigma * Q[s_n, a_n] + (1-sigma) * np.dot(Q[s_n], getSelectionProb(s_n, epsilon=0)))
                td_error = td_target - Q[s, a]
                z[s, a] += 1
                Q += lr * td_error * z
                z = gamma * lmbda  * z * (sigma + (1 - sigma) * getSelectionProb(s_n, epsilon=0)[a_n])
                s = s_n
                a = a_n
                sigma = math.log(1+sigma, 2)
                if done:
                    # print(str(epi_num) + " ended in " + str(epi_len) + " steps. Total reward: " + str(episode_reward))
                    stats_episode_length.append(epi_len)
                    #total episode return after that episode
                    stats_episode_reward.append(episode_reward)
                    break
    #Average over all returns with same lr
    avg_reward_list.append(np.average(stats_episode_reward))
write_data = ','.join(str(x) for x in avg_reward_list)
f.write("log decay sigma per step"+',')
f.write(write_data+'\n')
#Plot all different lr for a fixed sigma
plt.plot(alphas, avg_reward_list, colours[colour_index], label = "log decay sigma per step")
colour_index += 1
# plt.plot(np.arange(episodes), stats_episode_reward)
f.close()
plt.legend()
plt.show()
