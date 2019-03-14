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

import sys
print(sys.version)
