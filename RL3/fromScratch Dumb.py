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

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    #env.render()
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)




