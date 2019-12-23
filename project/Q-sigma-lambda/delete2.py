#Exponential decay

import numpy as np
import random
from windy_gridworld2 import WindyGridworldEnv, StochasticWindyGridworldEnv
import itertools
import matplotlib.pyplot as plt
import csv


np.random.seed(198)
env = StochasticWindyGridworldEnv()

episodes = 100
num_run = 100
# epsilon = 0.1
lr = 0.8
gamma = 1
lmbda = 0.6


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
#write header
avg_reward_list = list()
stats_episode_length = list()
epi_avg_list = list()
overall_per_episode_reward_list = list()


for run in range(num_run):
    print(run)
    Q=np.zeros((env.observation_space.n, env.action_space.n))
    stats_episode_reward = list()
    sigma = 1
    for epi_num in range(episodes):
        z=np.zeros((env.observation_space.n, env.action_space.n))
        s = env.reset()
        a = getEpsilonGreedyAction(s, epsilon=0.1)
        episode_reward = 0
        td_error_list_this_episode = []
        sigm_list_this_epiosed=[]
        temp_sigma=1
        last_sigma=1
        for epi_len in itertools.count():
            # env.render()
            # print(epi_num, s, a)
            s_n, reward, done, _ = env.step(a)
            episode_reward += reward
            a_n = getEpsilonGreedyAction(s_n, epsilon=0.1)
            td_target = reward + gamma * (temp_sigma * Q[s_n, a_n] + (1-temp_sigma) * np.dot(Q[s_n], getSelectionProb(s_n, epsilon=0)))
            td_error = td_target - Q[s, a]
            td_error_list_this_episode.append(abs(td_error))
            z[s, a] += 1
            Q += lr * td_error * z
            z = gamma * lmbda  * z * (temp_sigma + (1 - temp_sigma) * getSelectionProb(s_n, epsilon=0)[a_n])
            s = s_n
            a = a_n
            if(epi_num>0):
                sigma=clamp(abs(td_error/td_max), 0, 1)
                temp_sigma = sigma*last_sigma
                sigm_list_this_epiosed.append(sigma)
                # print(sigma)
            if(epi_len>4000):
                done=True
            if done:
                td_max = np.max(td_error_list_this_episode)
                stats_episode_reward.append(episode_reward)
                if(epi_num>0):
                    last_sigma = np.mean(sigm_list_this_epiosed)
                break
    overall_per_episode_reward_list.append(stats_episode_reward)
    epi_avg_list.append(np.sum(episode_reward))


overall_per_episode_reward_list = np.array(overall_per_episode_reward_list)
avg_overall_per_episode_reward_list = np.mean(overall_per_episode_reward_list, axis=0)
print(str(list(avg_overall_per_episode_reward_list)))
std_error = np.std(overall_per_episode_reward_list, axis = 0)/np.sqrt(num_run)
print("Std Error = ", str(list(std_error)))

print("Average = ", np.mean(avg_overall_per_episode_reward_list))
print("Average last few = ", np.mean(avg_overall_per_episode_reward_list[-100:]))
print("Average again =",epi_avg_list)

temp_sum = np.sum(overall_per_episode_reward_list, axis=1)
print("Average over entire return = ", np.mean(temp_sum))
print("SE over entire return = ", np.std(temp_sum)/np.sqrt(num_run))

#Plot all different lr for a fixed sigma
avg_overall_per_episode_reward_list=avg_overall_per_episode_reward_list
plt.plot(np.arange(len(avg_overall_per_episode_reward_list)), avg_overall_per_episode_reward_list, colours[0], label = "TD Error Sigma")

plt.legend()
plt.show()
