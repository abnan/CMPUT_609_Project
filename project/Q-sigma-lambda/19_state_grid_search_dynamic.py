import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import csv

class RandomWalk():
    def __init__(self):
        self.action_space_n = 2
        self.observation_space_n = 19
        self.STATES = np.arange(1, self.observation_space_n + 1)
        self.START_STATE = 10
        self.END_STATES = [0, self.observation_space_n + 1]
        self.state = self.START_STATE

    def step(self, action):
        if(self.state in self.END_STATES):
            return self.state, 0, True, "no info"

        if(action==0):
            self.state-=1
        else:
            self.state+=1

        if(self.state==0):
            done=True
            reward=-1
        elif(self.state==20):
            done=True
            reward=1
        else:
            done=False
            reward=0
        return self.state, reward, done, "no info"

    def reset(self):
        self.N_STATES = 19
        self.STATES = np.arange(1, self.N_STATES + 1)
        self.START_STATE = 10
        self.END_STATES = [0, self.N_STATES + 1]
        self.state = self.START_STATE
        return self.state



np.random.seed(98)
env = RandomWalk()

episodes = 50
num_run = 100
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma = 1
lmbdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0

for lmbda in lmbdas:
    for lr in alphas:
        RMS_error_per_run = np.zeros((num_run, episodes))
        for run in range(num_run):
            Q=np.zeros((env.observation_space_n+2, env.action_space_n))
            sigma = 0.95
            RMSError_list=[]
            for epi_num in range(episodes):
                z=np.zeros((env.observation_space_n+2, env.action_space_n))
                # print(lr, run, epi_num)
                s = env.reset()
                a = np.random.binomial(1, 0.5)
                episode_reward = 0
                for epi_len in itertools.count():
                    # env.render()
                    s_n, reward, done, _ = env.step(a)
                    # print(s_n, reward, done)
                    episode_reward += reward
                    a_n = np.random.binomial(1, 0.5)
                    # sigma = (1/np.e)**(epi_num)
                    # sigma = 1/(epi_num+1)**2
                    # print(sigma)
                    td_target = reward + gamma * (sigma * Q[s_n, a_n] + (1-sigma) * np.dot(Q[s_n], [0.5,0.5]))
                    td_error = td_target - Q[s, a]
                    z[s, a] += 1
                    Q += lr * td_error * z
                    z = gamma * lmbda  * z * (sigma + (1 - sigma) * [0.5, 0.5][a_n])
                    s = s_n
                    a = a_n
                    if done:
                        sigma=sigma*0.95
                        RMSError=0
                        AvgQ = np.zeros((env.observation_space_n+2))
                        for i in range(len(Q)):
                            AvgQ[i]=(Q[i][0]+Q[i][1])/2
                        RMSError = np.sum(np.square(AvgQ-TRUE_VALUE))
                        RMSError_list.append(np.sqrt(RMSError)/len(AvgQ))
                        break
            RMS_error_per_run[run]=RMSError_list

        overall_RMS_error=np.mean(RMS_error_per_run, axis=0)
        print("Lambda=", lmbda, "LR=", lr, np.mean(overall_RMS_error))
        area = np.trapz(overall_RMS_error)
        print("area =", area)
    # plt.plot(np.arange(len(overall_RMS_error)), overall_RMS_error)
    # plt.show()