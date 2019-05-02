import gym
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

#converting to simple concatenated string like [1,2,3,4,5]=12345
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf) (I did not check that these were good values)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf) (I did not check that these were good values)

    def transform(self, observation):
    # returns an int
      #  print(observation)
        cart_pos, cart_vel, pole_angle, pole_vel = observation

        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins),
        ])


class Q_learn:
    def __init__(self,n_s,n_a,epsilon=0.1,alpha=0.01,gamma=0.5):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.oldAction=None
        self.oldState=None
        self.Q = np.zeros((n_s,n_a))
        self.actions = np.arange(n_a)

    def getQ(self,state,action):
        return self.Q[state,action]
    
    def get_action(self,state,eps):
        self.oldState=state
        if np.random.random() <eps:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q[state,:])
        self.oldAction=action
        return action
    
    def learn(self,newstae,reward):
        if self.oldState ==None:
            return
        oldq = self.getQ(self.oldState,self.oldAction)
        maxQ =self.Q[newstae,:].max()
        self.Q[self.oldState,self.oldAction]=oldq+self.alpha*(reward+self.gamma*maxQ-oldq)


def play_one(model,ft ,eps):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        states=ft.transform(observation)
        action = model.get_action(states,eps)
       # prev_observation = observation
        observation, reward, done, _ = env.step(action)

        totalreward += reward

        if done and iters < 199:
            reward = -300

        # update the model
        #G = reward + gamma*np.max(model.predict(observation))
        #model.update(prev_observation, action, G)
        states=ft.transform(observation)
        model.learn(states,reward)
        iters += 1

    return totalreward


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__=='__main__':
    env = gym.make('CartPole-v0').env
    ft = FeatureTransformer()
    n_states=10**env.observation_space.shape[0]
    n_actions=env.action_space.n
    model = Q_learn(n_states,n_actions)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    
    N=10000
    totalrewards=np.empty(10000)

    for n in range(N):
        eps= 1.0/np.sqrt(n+1)
        totalreward = play_one(model,ft,eps)
        totalrewards[n]=totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
        
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
        




        