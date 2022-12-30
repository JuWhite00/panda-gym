# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:24:49 2022
@author: tangu
"""
import gym
from panda_gym.pybullet import PyBullet as p 
from panda_gym.envs.robots.doosan import Doosan
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.panda_tasks.doosan_task import DoosanTest
import numpy as np
import time
import hydra




env = DoosanTest()

observation = env.reset()
#action = np.array([-0.98, 0.95, -0.9,0.6,-0.8,0.1,0.4])
for _ in range(1000):
     # random action
    pose_target_object = env.get_pos_obs('obj')
    action = pose_target_object[0]
    print("pose: ", pose_target_object[1])
    
    observation, reward, done, info = env.step(action)
    
    time.sleep(0.03)


env.close()