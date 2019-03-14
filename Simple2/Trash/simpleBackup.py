"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19

"""

from uofgsocsai import *
import gym
import numpy as np
import time
import networkx as nx


from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import lines

from helpers import *

AIMA_TOOLBOX_ROOT="/home/neil/AI/aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *

def enviornment():
    problem_id = 1
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=0.0)
    print(env.desc)
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
    maze_map = UndirectedGraph(state_space_actions)
    G = nx.Graph()
    node_labels = dict()
    node_colors = dict()
    for n, p in state_space_locations.items(): 
        G.add_node(n)  
        node_labels[n] = n  
        node_colors[n] = "white"  
    initial_node_colors = dict(node_colors)
    node_label_pos = {k: [v[0], v[1] - 0.25] for k, v in
                      state_space_locations.items()}  
    edge_labels = dict()
    for node in maze_map.nodes():
        connections = maze_map.get(node)
        for connection in connections.keys():
            distance = connections[connection]
            G.add_edge(node, connection)
            edge_labels[(node, connection)] = distance 
    print("Done creating the graph object")
    return maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos

def enviornment_eval(problem_id):
    problem_id = problem_id
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=0.0)
    print(env.desc)
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
    maze_map = UndirectedGraph(state_space_actions)
    G = nx.Graph()

    node_labels = dict()
    node_colors = dict()

    for n, p in state_space_locations.items():  
        G.add_node(n)  
        node_labels[n] = n  
        node_colors[n] = "white" 
 
    initial_node_colors = dict(node_colors)
    node_label_pos = {k: [v[0], v[1] - 0.25] for k, v in
                      state_space_locations.items()} 
    edge_labels = dict()

    for node in maze_map.nodes():
        connections = maze_map.get(node)
        for connection in connections.keys():
            distance = connections[connection]
            G.add_edge(node, connection) 
            edge_labels[(node, connection)] = distance  
    print("Done creating the graph object")
    return maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos


def mazeproblem(state_initial_id, state_goal_id, maze_map):
    maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)

    print("Initial state: " + maze_problem.initial)
    print("Goal state: " + maze_problem.goal)
    return maze_problem


def my_best_first_graph_search(problem, f, initial_node_colors):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
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
        return (iterations, all_node_colors, node)

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
            return (iterations, all_node_colors, node)

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


def my_astar_search(initial_node_colors, problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')  # define heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n), initial_node_colors)


def final_path_colors(problem, solution, initial_node_colors):
    "returns a node_colors dict of the final path provided the problem and solution"
    final_colors = dict(initial_node_colors)
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors


def main(state_initial_id, maze_problem, initial_node_colors):
    all_node_colors = []
    iterations, all_node_colors, node = my_astar_search(initial_node_colors, problem=maze_problem, h=None)
    print(iterations)
    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)
    while cnode.state != state_initial_id:
        cnode = cnode.parent
        solution_path.append(cnode)

    print("----------------------------------------")
    print("Identified goal state:" + str(solution_path[0]))
    print("----------------------------------------")
    print("Solution trace:" + str(solution_path))
    print("----------------------------------------")
    print("Final solution path:")
    i = 0
    for node in solution_path:
        i+=1
    print("Number of steps in optimal solution = " + str(i))
    return solution_path,iterations
    


# show_map(final_path_colors(maze_problem, node.solution()))


def filegenerate(problem_id, solution_path):
    filename = "out_simple_" + str(problem_id)
    text = "Identified goal state:" + str(solution_path[0]) + "\n" + "Solution trace:" + str(solution_path)
    print(str(text))
    with open(filename + ".txt", "w") as file:
        file.write(str(text))



maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos = enviornment()
maze_problem = mazeproblem(state_initial_id, state_goal_id, maze_map)
solution_path,iterations = main(state_initial_id, maze_problem, initial_node_colors)
filegenerate(problem_id, solution_path)


