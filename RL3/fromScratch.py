"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import sys
import matplotlib.pyplot as plt
AIMA_TOOLBOX_ROOT="/home/neil/AI/aima-python" #Required on my machine
sys.path.append(AIMA_TOOLBOX_ROOT)            #Required on my machine
from rl import *
from mdp import sequential_decision_environment
from mdp import value_iteration
import time
from uofgsocsai import *

import sys
print(sys.version)
import gym
import numpy as np
import tensorflow as tf
import argparse
import random
from IPython.display import clear_output

#Set up environment

problem_id=0
env = LochLomondEnv(problem_id=problem_id,is_stochastic=False,reward_hole=0.0)

#env.render()

#print("Action Space {}".format(env.action_space))
#print("State Space {}".format(env.observation_space))

env.s =2

#env.render()

#Settinig up the Reward Dable

#print(env.P[57])
#{action: [(probability, nextstate, reward, done)]}



env.s = 0  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

#initialize q-table of zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

#-----------------Training the Agent-------------------

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 1000):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        if done and reward:
            print("Done " + str(done))
            print("info " + str(info))
            print("reward " + str(reward))
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

#-----------------End Training the Agent-------------------

print(q_table[1])


