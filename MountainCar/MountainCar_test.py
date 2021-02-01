import gym
import numpy as np

# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
env._max_episode_steps = 5000
env.reset()

# take 1000 random actions
for _ in range(1000):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample())
    print("DEBUG: state, reward, done, info = ", state, reward, done, info)
env.close()

# follow a hard-coded policy
for _ in range(50):
    env.render()
    for left in range(34):
        state, reward, done, info = env.step(0)
        print("DEBUG: state, reward, done, info = ", state, reward, done, info)
    for right in range(34):
        state, reward, done, info = env.step(2)
        print("DEBUG: state, reward, done, info = ", state, reward, done, info)
env.close()