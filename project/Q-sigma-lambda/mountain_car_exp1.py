import numpy as np
import tilecoding

# possible actions
ACTIONS = np.arange(3)

# bounds for position and velocity
POSITION_MIN = - 1.2
POSITION_MAX = 0.5
VELOCITY_MIN = - 0.07
VELOCITY_MAX = 0.07

# get action values
def getValue(state, weights, hash_table, n_tilings):
    Q = np.zeros(3)
    # for each action
    # get tile indices and compute Q value as sum of all active tiles' weights
    for i in ACTIONS:
        active_tiles = getActiveTiles(state[0], state[1], i, hash_table, n_tilings)
        Q[i] = np.sum(weights[active_tiles])
    return Q

# preprocess state: scale position and velocity
def preprocessState(state, n_tilings):
    position = state[0]
    velocity = state[1]
    # scale state (position, velocity)
    position_scale = n_tilings / (POSITION_MAX - POSITION_MIN)
    velocity_scale = n_tilings / (VELOCITY_MAX - VELOCITY_MIN)
    position = position_scale * position # - POSITION_MIN)
    velocity = velocity_scale * velocity #- VELOCITY_MIN)
    return np.array((position, velocity))

# get active tile for each tiling
def getActiveTiles(position, velocity, action, hash_table, n_tilings):
    active_tiles = tilecoding.tiles(hash_table, n_tilings, 
                                    [position, velocity], [action])
    return active_tiles

# get number of steps to reach the goal under current state value function
def costToGo(state, weights, hash_table, n_tilings):
    costs = []
    for action in ACTIONS:
        costs.append(getValue(state, weights, hash_table, n_tilings))
    return - np.max(costs)

def qSigmaMC(env,  n_episodes = 100, Lambda = 0, sigma = 1, epsilon = 0.1, alpha = 0.1, gamma = 1, 
                   target_policy = "greedy", printing = False, 
                   cliff = False, n_tilings = 8, max_size = 4096, render = False,
                   update_sigma = 1): 
    
    # adjust learning rate to number of tilings
    alpha = alpha / n_tilings
    hash_table = tilecoding.IHT(max_size)
    
    if target_policy == "greedy":
        epsilon_target = 0
    weights = np.zeros(max_size)
    episode_steps = np.zeros(n_episodes)
    returns = np.zeros(n_episodes)
    
    for i in range(n_episodes):
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
            
            # only for Mountain Cliff, negative reward of -100 when falling of the cliff
            if cliff:
                if s_n[0] <= POSITION_MIN:
                    s_n = env.reset()
                    r = - 100
                    
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
            active_tiles = getActiveTiles(s[0], s[1], a, hash_table, n_tilings)
            # update eligibility for all active tiles
            E[active_tiles] = E[active_tiles] * (1 - beta) + 1
            
            # update weights
            weights += alpha * E * td_error
            
            # reduce eligibility for all weights
            E *= gamma * Lambda * (sigma + policy[a_n] * (1 - sigma))
            
            # set s to s_n, a to a_n, Q to Q_n
            s = s_n
            a = a_n
            Q = Q_n
            
            if done:
                if printing:
                    print("Episode " + str(i + 1) + " finished after " + 
                          str(j + 1) + " time steps " + "obtaining " + 
                          str(reward_sum) + " returns.")
                episode_steps[i] = j
                returns[i] = reward_sum
                sigma *= update_sigma
                break
    return episode_steps, returns, weights, None
