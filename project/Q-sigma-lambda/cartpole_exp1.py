#Exponential decay

import numpy as np
import random
import tilecoding
import itertools
import matplotlib.pyplot as plt
import csv
import gym

# np.random.seed(98)

# possible actions
ACTIONS = np.arange(2)

# bounds for position and velocity
POSITION_MIN = -4.8
POSITION_MAX = 4.8
POLE_ANGLE_MIN = -0.418
POLE_ANGLE_MAX = 0.418


def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n
def getPolicy(Q, epsilon):
    """
    Get probabilities of epsilon-greedy policy with respect to Q. 
    
    Parameters
    ----------
    Q: numpy array (float vector)
        Action value function
    epsilon: float(1) in [0, 1]
        Choice to sample a random action.
    
    Return
    ------
    A policy of same length as Q specifying the action probabilities.
    """
    greedy_action = np.argmax(Q)
    policy = np.repeat(epsilon / len(Q), len(Q))
    policy[greedy_action] += 1 - epsilon
    return policy
    
def sampleAction(policy):
    """
    Sample action from policy.
    
    Parameters
    ----------
    policy: numpy array
        The policy, a valid probability distribution.
        
    Return
    ------
    An action, a single integer value.
    """
    return np.random.choice(np.arange(0, len(policy)), p = policy)


# get action values
def getValue(state, weights, hash_table, n_tilings):
    Q = np.zeros(2)
    # for each action
    # get tile indices and compute Q value as sum of all active tiles' weights
    for i in ACTIONS:
        active_tiles = getActiveTiles(state[0], state[1], state[2], state[3], i, hash_table, n_tilings)
        Q[i] = np.sum(weights[active_tiles])
    return Q

# preprocess state: scale position and velocity
def preprocessState(state, n_tilings):
    position = state[0]
    velocity = state[1]
    pole_angle = state[2]
    pole_velocity = state[3]
    # print(position, velocity, pole_angle, pole_velocity)
    # scale state (position, velocity)
    position_scale = 10 / (POSITION_MAX - POSITION_MIN)
    pole_angle_scale = 10/(POLE_ANGLE_MAX - POLE_ANGLE_MIN)
    position = position_scale * position 
    pole_angle = pole_angle_scale * pole_angle
    
    return np.array((position, velocity, pole_angle, pole_velocity))

# get active tile for each tiling
def getActiveTiles(position, velocity, pole_angle, pole_velocity, action, hash_table, n_tilings):
    active_tiles = tilecoding.tiles(hash_table, n_tilings, 
                                    [position, velocity, pole_angle, pole_velocity], [action])
    return active_tiles

# get number of steps to reach the goal under current state value function
def costToGo(state, weights, hash_table, n_tilings):
    costs = []
    for action in ACTIONS:
        costs.append(getValue(state, weights, hash_table, n_tilings))
    return - np.max(costs)

def qSigmaMC(env,  n_episodes = 100, Lambda = 0, sigma = 1, epsilon = 0.1, alpha = 0.1, gamma = 1, 
                   target_policy = "greedy", printing = False, n_tilings = 8, max_size = 4096, render = False): 
    
    # adjust learning rate to number of tilings
    alpha = alpha / n_tilings
    hash_table = tilecoding.IHT(max_size)
    
    if target_policy == "greedy":
        epsilon_target = 0
    weights = np.zeros(max_size)
    episode_steps = np.zeros(n_episodes)
    returns = np.zeros(n_episodes)
    
    for i in range(n_episodes):
        # print(i)
        done = False
        j = 0
        reward_sum = 0
        # at begin of episode: initialize eligibility for each weight to 0
        E = np.zeros_like(weights)
        # get initial state and scale this state
        s = env.reset()
        s = preprocessState(s, n_tilings)

        # get action values
        Q = getValue(s, weights, hash_table, n_tilings)
        # get action probabilities (epsilon-greedy behavior policy)
        policy = getPolicy(Q, epsilon)
        # sample action from policy
        a = sampleAction(policy)
        while done == False:
            j += 1
            # take action, observe next state and reward
            if render:
                env.render()
            s_n, r, done, _ = env.step(a)
            
                    
            reward_sum += r
            
            # sample next action according to new state
            s_n = preprocessState(s_n, n_tilings)
            Q_n = getValue(s_n, weights, hash_table, n_tilings)
            policy = getPolicy(Q_n, epsilon)
            a_n = sampleAction(policy)
            
            if target_policy == "greedy":
                policy = getPolicy(Q_n, epsilon_target)
            
            # compute td target and td error
            sarsa_target = Q_n[a_n]
            exp_sarsa_target = np.dot(policy, Q_n)
            td_target = r + gamma * (sigma * sarsa_target + 
                                     (1 - sigma) * exp_sarsa_target)
            td_error = td_target - Q[a]
            # get active tiles
            active_tiles = getActiveTiles(s[0], s[1], s[2], s[3], a, hash_table, n_tilings)
            # update eligibility for all active tiles
            E[active_tiles] += 1
            
            # update weights
            weights += alpha * E * td_error
            
            # reduce eligibility for all weights
            E *= gamma * Lambda * (sigma + policy[a_n] * (1 - sigma))
            
            # set s to s_n, a to a_n, Q to Q_n
            s = s_n
            a = a_n
            Q = Q_n
            # if(j>2000):
            #     done = True
                # print("exceeded")
            if done:
                if printing:
                    print("Episode " + str(i + 1) + " finished after " + 
                          str(j + 1) + " time steps " + "obtaining " + 
                          str(reward_sum) + " returns.")
                episode_steps[i] = j
                returns[i] = reward_sum
                if(i in returns_dict):
                    returns_dict[i].append(reward_sum)
                else:
                    returns_dict[i] = [reward_sum]
                sigma *=0.95
                break
    return episode_steps, returns, weights, None


num_episodes = 50
returns_dict = dict()
env = gym.make("CartPole-v0").env
num_run = 10000

for n_run in range(num_run):
    print(n_run)
    episode_steps, returns, weights, _ = qSigmaMC(env, max_size=10000, n_episodes = num_episodes, sigma=0.95, Lambda=0.7, alpha=0.5, render=False, printing=False)

avg_rewards = []
std_error = []
for i in range(0, num_episodes):
    avg_rewards.append(np.mean(returns_dict[i]))
    std_error.append(np.std(returns_dict[i])/np.sqrt(num_run))
print("Average reward=",np.mean(avg_rewards))
area = np.trapz(avg_rewards)
print("area =", area)
rc_moving_avg_rewards = []
# for i in range(10, len(avg_rewards)):
#     rc_moving_avg_rewards.append(np.mean(avg_rewards[i-10:i]))
# plt.plot(np.arange(len(rc_moving_avg_rewards)), rc_moving_avg_rewards, label = "Dynamic decay sigma; lambda=0.5")
plt.plot(np.arange(len(avg_rewards)), avg_rewards, label = "Dynamic decay sigma; lambda=0.5")
plt.xlabel('episodes')
plt.ylabel('average return')
plt.legend()
plt.show()
# print(rc_moving_avg_rewards)
print(avg_rewards)
print("===========================================================================")
print(std_error)
