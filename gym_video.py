import gym
from gym import wrappers

env = gym.make('CartPole-v0')

env = wrappers.Monitor(env,"fldr1")

obser = env.reset()
done =False
while not done:
    action = env.action_space.sample()
    obse,rew,done,info=env.step(action)

    if done:
        break
