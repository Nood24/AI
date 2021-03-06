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

print("Initial state: " + maze_problem.initial)
print("Goal state: "    + maze_problem.goal)

#------------------------A-Star-Search-start----------------

def my_best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = dict(initial_node_colors)
    
    f = memoize(f, 'f')
    node = Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:
        node = frontier.pop()
        
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None



def my_astar_search(problem, h=0):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))
#-----------------------A-Star-Search-End------------------

#----------------------Runner End-----------------------
all_node_colors=[]
iterations, all_node_colors, node = my_astar_search(problem=maze_problem, h=None)

#-- Trace the solution --#
solution_path = [node]
cnode = node.parent
solution_path.append(cnode)
while cnode.state != "S_00_00":    
    cnode = cnode.parent  
    solution_path.append(cnode)

#-------------------------Runner End---------------

print("----------------------------------------")
print("Identified goal state:"+str(solution_path[0]))
print("----------------------------------------")
print("Solution trace:"+str(solution_path))
print("----------------------------------------")
print("Final solution path:")


# Reset the random generator to a known state (for reproducability)
np.random.seed(12)

####
for e in range(max_episodes): # iterate over episodes
    observation = env.reset() # reset the state of the env to the starting state     
    
    for iter in range(max_iter_per_episode):
      #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line   


   
      action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
      observation, reward, done, info = env.step(action) # observe what happends when you take the action
      
      # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner

      print("e,iter,reward,done =" + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done))

      # Check if we are done and monitor rewards etc...
      if(done and reward==reward_hole): 
          env.render()     
          print("We have reached a hole :-( [we can't move so stop trying; just give up]")
          break

      if (done and reward == +1.0):
          env.render()     
          print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
          break
     
      

