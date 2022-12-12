# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:24:49 2022

@author: tangu
"""
# import gym
# from panda_gym.pybullet import PyBullet
# from panda_gym.envs.robots.doosan import Doosan
# from panda_gym.envs.robots.panda import Panda
# from panda_gym.envs.panda_tasks.doosan_task import DoosanTest
# import numpy as np
# import time
import os
import random
from sys import platform

dirname = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0])[0]

if platform == "win32":
    path_obj = dirname + '\\mesh\\pybullet-URDF-models\\urdf_models\\models\\'
    select_name = random.sample(os.listdir(path_obj),1)[0]
    path_obj = path_obj + select_name + '\\model.urdf'
else:
    path_obj = dirname + '/mesh/pybullet-URDF-models/urdf_models/models/'
    select_name = random.sample(os.listdir(path_obj),1)[0]
    path_obj = path_obj + select_name + '/model.urdf'


# env = DoosanTest(render= True)

# observation = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample() # random action
#     observation, reward, done, info = env.step(action)
    
#     time.sleep(0.03)


# env.close()