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
import os, sys
import tensorflow as tf

from helpers import *
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)


problem_id = 0       
reward_hole = 0.0    
is_stochastic = True  

max_episodes = 200  
max_iter_per_episode = 500 

env = LochLomondEnv(problem_id=problem_id, is_stochastic=True,   reward_hole=reward_hole)



inputs = tf.placeholder(shape=[1,16],dtype=tf.float32)


weights = tf.Variable(tf.random_uniform([16,4],0,0.1))
 
Q1 = tf.matmul(inputs,weights)

output = tf.argmax(Q1,1)

Q2 = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q2 - Q1))
gdo = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updatedweights = gdo.minimize(loss)

gamma = 0.9
epsilon = 0.1
episodes = 2000

totalReward = 0

session = tf.Session()
session.run(tf.initialize_all_variables())
for i in range(episodes):
    state_now = env.reset()
    done = False
    reward = 0
    for j in range(100):
        #Lets find the dot product of the inputs with the weights and apply argmax on it.
        action , Y = session.run([output, Q1], feed_dict = {inputs : [np.eye(16)[state_now]]})
        if epsilon > np.random.rand(1):
            action[0] = env.action_space.sample()
            epsilon -= 10**-3
        #Lets iterate to the next state Note: This can be random.
        state_next , reward, done, _ = env.step(action[0])
        Y1 = session.run(Q1, feed_dict = {inputs : [np.eye(16)[state_next]]})
        change_Y = Y
        change_Y[0, action[0]] = reward + gamma*np.max(Y1)
        #Updating the weights 
        _,new_weights = session.run([updatedweights,weights],feed_dict={inputs:[np.eye(16)[state_now]],Q2:change_Y})
        #Lets append the total number of rewards
        totalReward += reward
        state_now = state_next
        if reward == 1:
            print ('Episode {} was successful, Agent reached the Goal'.format(i))
        
       
session.close()









