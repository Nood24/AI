"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv 
import os, sys
from helpers import *
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)


problem_id = 0       
reward_hole = 0.0   
is_stochastic = True  
max_episodes = 200   
max_iter_per_episode = 500 


env = LochLomondEnv(problem_id=problem_id, is_stochastic=True,   reward_hole=reward_hole)
print(env.desc)

state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

np.random.seed(12)

wins = 0
losses = 0
totalMoves = 0


for e in range(max_episodes): 
    observation = env.reset()
    
    for iter in range(max_iter_per_episode):
      env.render()  #Uncomment for debugging
      action = env.action_space.sample() 
      observation, reward, done, info = env.step(action) 

      # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner 

      totalMoves+=1

      print("e,iter,reward,done =" + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done))

      if(done and reward==reward_hole): 
          env.render()     
          print("We have reached a hole :-( [we can't move so stop trying; just give up]")
          losses+=1
          break

      if (done and reward == +1.0):
          env.render()     
          print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
          wins+=1
          break

print ("Wins: " + str(wins))
print ("Losses: " + str(losses))
if (losses > 0):
     print ("win Rate: " + str(wins/losses))
else:
     print ("win Rate: 0")
print("Average moves: " + str(totalMoves/max_episodes))
     
      

