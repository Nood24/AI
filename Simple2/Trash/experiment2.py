"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem

#-------------------------------Added-------------------------
import os, sys
from helpers import *

import matplotlib.pyplot as plt
from matplotlib import lines
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
import networkx as nx


AIMA_TOOLBOX_ROOT="/home/neil/AI/aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *


#------------------------------End Add--------------------------



# Setup the parameters for the specific problem (you can change all of these if you want to) 
problem_id = 0        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

max_episodes = 2   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_iter_per_episode = 2 # you decide how many iterations/actions can be executed per episode

# Generate the specific problem 
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)

# Let's visualize the problem/env
print(env.desc)

# Create a representation of the state space for use with AIMA A-star
state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)


print("state_space_locations:\n\n" + str(state_space_locations))
print("state_space_actions:\n\n" + str(state_space_actions))
print("state_initial_id:\n\n" + str(state_initial_id))

maze_problem = GraphProblem('state_initial_id', 'state_goal_id', state_space_locations)

#ASTAR SEARCH-----------------------------------------------------


#-- Trace the solution --#
solution_path = [node]
cnode = node.parent
solution_path.append(cnode)
while cnode.state != "S_00_00":    
    cnode = cnode.parent  
    solution_path.append(cnode)

print("----------------------------------------")
print("Identified goal state:"+str(solution_path[0]))
print("----------------------------------------")
print("Solution trace:"+str(solution_path))
print("----------------------------------------")
print("Final solution path:")
show_map(final_path_colors(maze_problem, node.solution()))





