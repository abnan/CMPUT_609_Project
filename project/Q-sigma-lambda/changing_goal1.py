#Exponential decay

import numpy as np
import random
from windy_gridworld2 import WindyGridworldEnv, StochasticWindyGridworldEnv
import itertools
import matplotlib.pyplot as plt
import csv


np.random.seed(98)
env = StochasticWindyGridworldEnv()

episodes = 100
num_run = 10
# epsilon = 0.1
# lr = 0.6
gamma = 1
# lmbda = 0.7


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
lmdbs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colour_index = 0
#write header
for lmbda in lmdbs:
    for lr in alpha:
        print("Lambda =", lmbda, "Alpha =", lr)
        avg_reward_list = list()
        stats_episode_length = list()
        epi_avg_list = list()
        overall_per_episode_reward_list = list()
        useless_list=[]
        for run in range(num_run):
            # print(run)
            Q=np.zeros((env.observation_space.n, env.action_space.n))
            stats_episode_reward = list()
            sigma = 0.95
            for epi_num in range(episodes):
                z=np.zeros((env.observation_space.n, env.action_space.n))
                s = env.reset()
                a = getEpsilonGreedyAction(s, epsilon=0.1)
                episode_reward = 0
                for epi_len in itertools.count():
                    # env.render()
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
                    if(epi_len>4000):
                        done=True
                    if done:
                        # print(str(epi_num) + " ended in " + str(epi_len) + " steps. Total reward: " + str(episode_reward))
                        stats_episode_length.append(epi_len)
                        #total episode return after that episode
                        stats_episode_reward.append(episode_reward)
                        sigma=sigma*0.95
                        break
            useless_list.append(np.mean(stats_episode_reward))
            overall_per_episode_reward_list.append(stats_episode_reward)
            epi_avg_list.append(np.mean(episode_reward))
        # print(overall_per_episode_reward_list)
        overall_per_episode_reward_list = np.array(overall_per_episode_reward_list)
        avg_overall_per_episode_reward_list = np.mean(overall_per_episode_reward_list, axis=0)
        # print(str(list(avg_overall_per_episode_reward_list)))
        std_error = np.std(overall_per_episode_reward_list, axis = 0)/np.sqrt(num_run)
        # print("Std Error = ", str(list(std_error)))
        print("Average = ", np.mean(avg_overall_per_episode_reward_list))
        # print("Average last few = ", np.mean(avg_overall_per_episode_reward_list[-100:]))

        #Plot all different lr for a fixed sigma
        # avg_overall_per_episode_reward_list=avg_overall_per_episode_reward_list[-100:]
        # plt.plot(np.arange(len(avg_overall_per_episode_reward_list)), avg_overall_per_episode_reward_list, colours[0], label = "Dynamic Decay Sigma")

        # plt.legend()
        # plt.show()
