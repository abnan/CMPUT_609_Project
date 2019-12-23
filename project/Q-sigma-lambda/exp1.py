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
num_run = 1000
# epsilon = 0.1
# lr = 0.6
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

f = open("results_exp1.csv", "w")
colours = ['r-','b-','g-','c-','y-','k-']
colour_index = 0
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
write_data = ','.join(str(x) for x in alphas)
#write header
f.write("sigma,"+write_data+"\n")
avg_reward_list = list()
for lr in alphas:
    print(lr)
    stats_episode_length = list()
    stats_episode_reward = list()
    epi_avg_list = list()
    for run in range(num_run):
        Q=np.zeros((env.observation_space.n, env.action_space.n))
        sigma = 0.99
        for epi_num in range(episodes):
            z=np.zeros((env.observation_space.n, env.action_space.n))
            # print(lr, run, epi_num)
            s = env.reset()
            a = getEpsilonGreedyAction(s, epsilon=0.1)
            episode_reward = 0
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
                z[s, a] += 1
                Q += lr * td_error * z
                z = gamma * lmbda  * z * (sigma + (1 - sigma) * getSelectionProb(s_n, epsilon=0)[a_n])
                s = s_n
                a = a_n
                if done:
                    # print(str(epi_num) + " ended in " + str(epi_len) + " steps. Total reward: " + str(episode_reward))
                    stats_episode_length.append(epi_len)
                    #total episode return after that episode
                    stats_episode_reward.append(episode_reward)
                    sigma=sigma*0.99
                    break
        epi_avg_list.append(np.mean(episode_reward))
    #Average over all returns with same lr
    avg_reward_list.append(np.average(stats_episode_reward))
    print("Std Error = ", np.std(epi_avg_list)/np.sqrt(num_run))
write_data = ','.join(str(x) for x in avg_reward_list)
f.write("Dynamic decay sigma 0.99; lambda=0.7"+',')
f.write(write_data+'\n')
#Plot all different lr for a fixed sigma
plt.plot(alphas, avg_reward_list, colours[colour_index], label = "Dynamic decay sigma 0.99; lambda=0.7")
# plt.plot(np.arange(episodes), stats_episode_reward)
f.close()
plt.legend()
plt.show()
