import gymnasium as gym
from numpngw import write_apng  # pip install numpngw or pip install panda-gym[extra]

import panda_gym

env = gym.make("PandaStack-v2", render=True)
images = []


obs, info = env.reset()
done = False
images.append(env.render("rgb_array"))

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    images.append(env.render("rgb_array"))

env.close()

write_apng("stack.png", images, delay=40)
