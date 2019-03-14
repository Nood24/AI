
from time import sleep
import numpy as np
import gym
import sys
import time, pickle, os
from uofgsocsai import *


def environment():
    # Environment
    problem_id=int(sys.argv[1])
    print(problem_id)
    # problem_id=0
    env = LochLomondEnv(problem_id=problem_id,is_stochastic=False,reward_hole=0.0)
    epsilon = 0.9
    total_episodes = 100
    max_steps = 100

    lr_rate = 0.81
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return env,problem_id,epsilon,total_episodes,max_steps,lr_rate,gamma,Q
def environment_eval(problem_id):
    # Environment
    problem_id=problem_id
    # problem_id=0
    env = LochLomondEnv(problem_id=problem_id,is_stochastic=False,reward_hole=0.0)

    epsilon = 0.9
    total_episodes = 100
    max_steps = 100

    lr_rate = 0.81
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return env,problem_id,epsilon,total_episodes,max_steps,lr_rate,gamma,Q


def choose_action(state,epsilon,env,Q):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action,Q,gamma,lr_rate):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

def main(total_episodes,env,max_steps,Q,gamma,lr_rate,epsilon):
    # Start
    reward_list=[]
    iter_list=[]
    for episode in range(total_episodes):
        state = env.reset()
        t = 0

        while t < max_steps:
            env.render()

            action = choose_action(state,epsilon,env,Q)

            state2, reward, done, info = env.step(action)

            reward_list.append(reward)
            if(done and reward == +1.0):
                iter_list.append(t)

            learn(state, state2, reward, action,Q,gamma,lr_rate)

            state = state2

            t += 1

            if done:
                break

            # time.sleep(0.1)

    print(Q)
    return reward_list,iter_list

def pklgenerate(Q):
    with open("frozenLake_qTable.pkl", 'wb') as f:
        pickle.dump(Q, f)

def filegenerate(problem_id,Q):
    filename= "out_rl_"+str(problem_id)
    text=Q
    print(str(text))
    with open(filename+".txt", "w") as file:
        file.write("left , right , down ,  up")
        file.write("\n")
        file.write(str(text))

if __name__ == '__main__':
    env,problem_id,epsilon,total_episodes,max_steps,lr_rate,gamma,Q=environment()
    main(total_episodes,env,max_steps,Q,gamma,lr_rate,epsilon)
    #pklgenerate(Q)
    filegenerate(problem_id,Q)

  
