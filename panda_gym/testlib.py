# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:24:49 2022

@author: tangu
"""
import gymnasium as gym
from panda_gym.pybullet_gym import PyBullet
from panda_gym.envs.robots.doosan import Doosan
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.panda_tasks import DoosanTest
import numpy as np

sim = PyBullet(render = False)
robot = Doosan(sim)


for _ in range(10000):
    robot.set_action(np.array([1.0]))
    sim.step()
    sim.render()
    
sim.__del__()

# env = gym.make("DoosanTest-v3", render=True)

# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample() # random action
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()