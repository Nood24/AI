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

#maze_problem = GraphProblem('state_initial_id', 'state_goal_id', state_space_locations)

#ASTAR SEARCH-----------------------------------------------------

# initialise a graph
G = nx.Graph()

# use this while labeling nodes in the map
node_labels = dict()
node_colors = dict()
for n, p in state_space_locations.items():
    G.add_node(n)            # add nodes from locations
    node_labels[n] = n       # add nodes to node_labels
    node_colors[n] = "white" # node_colors to color nodes while exploring the map

# we'll save the initial node colors to a dict for later use
initial_node_colors = dict(node_colors)
    
# positions for node labels
node_label_pos = {k:[v[0],v[1]-0.25] for k,v in state_space_locations.items()} # spec the position of the labels relative to the nodes

# use this while labeling edges
edge_labels = dict()

# add edges between nodes in the map - UndirectedGraph defined in search.py
for node in state_space_actions:
    print (node)
    connections = state_space_actions
    for connection in connections.keys():
        distance = connections[connection]        
        G.add_edge(node, connection) # add edges to the graph        
        edge_labels[(node, connection)] = distance # add distances to edge_labels
        
print("Done creating the graph object")

def show_map(node_colors):
    # set the size of the plot
    plt.figure(figsize=(16,13))
    # draw the graph (both nodes and edges) with locations
    nx.draw(G, pos = state_space_locations, node_color = [node_colors[node] for node in G.nodes()])

    # draw labels for nodes
    node_label_handles = nx.draw_networkx_labels(G, pos = node_label_pos, labels = node_labels, font_size = 9)
    # add a white bounding box behind the node labels
    [label.set_bbox(dict(facecolor='white', edgecolor=None)) for label in node_label_handles.values()]

    # add edge lables to the graph
    nx.draw_networkx_edge_labels(G, pos = state_space_locations, edge_labels=edge_labels, font_size = 8)
    
    # add a legend
    white_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="white")
    orange_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="orange")
    red_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="red")
    gray_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="gray")
    green_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="green")
    plt.legend((white_circle, orange_circle, red_circle, gray_circle,green_circle),
               ('Un-explored', 'Frontier', 'Currently exploring', 'Explored', 'Solution path'),
               numpoints=1,prop={'size':16}, loc=(.8,1.0))

maze_problem = GraphProblem('state_initial_id', 'state_goal_id', state_space_locations)

print("Initial state: " + maze_problem.initial)
print("Goal state: "    + maze_problem.goal)

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
        for child in node:
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

def my_astar_search(problem, h= 0):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))
    

def final_path_colors(problem, solution):
    "returns a node_colors dict of the final path provided the problem and solution"
    
    # get initial node colors
    final_colors = dict(initial_node_colors)
    # color all the nodes in solution and starting node to green
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"  
    return final_colors

    
def display_visual(user_input, algorithm=None, problem=None):
    if user_input == False:
        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(all_node_colors[iteration])
            except:
                pass
        def visualize_callback(Visualize):
            if Visualize is True:
                button.value = False
                
                global all_node_colors
                
                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(problem, solution))
                
                slider.max = len(all_node_colors) - 1
                
                for i in range(slider.max + 1):
                    slider.value = i
                    #time.sleep(3.)
        
        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration = slider)
        display(slider_visual)

        button = widgets.ToggleButton(value = False)
        button_visual = widgets.interactive(visualize_callback, Visualize = button)
        display(button_visual)

all_node_colors=[]
iterations, all_node_colors, node = my_astar_search(problem=maze_problem, h=0)

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





