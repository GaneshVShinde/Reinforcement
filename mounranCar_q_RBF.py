#%%

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor



#%%
class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components))
#                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print "observations:", observations
       # print(observations)
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)
    
    

#%%
env = gym.make('MountainCar-v0')
data = [env.observation_space.sample() for i in range(10000)]

ft = FeatureTransformer(env)
#model = Model(env, ft, "constant")
transformed=ft.transform(data)


  #%%

def get_reset_sample():
    return ft.transform([env.reset()])

class Q_model:
    def __init__(self,lr,n_actions,gamma=0.7,sample_func=get_reset_sample):
        self.models = []
        self.n_actions=n_actions
        self.gamma = gamma
        self.oldState = None
        self.oldAction = None
        for _ in range(n_actions):
            model= SGDRegressor(learning_rate=lr)
            model.partial_fit(sample_func(), [0])
            self.models.append(model)
    
    def predict(self,state):
        return np.stack([m.predict(state) for m in self.models])
    
    def learn(self,state,reward):
        if self.oldAction == None:
            return
        newq=self.predict(state)
        G = reward+self.gamma*np.max(newq)
        self.models[self.oldAction].partial_fit(self.oldState,[G])
    
    def get_action(self,state,eps):
        self.oldState=state
        if np.random.random()<eps:
            action= np.random.choice(np.arange(self.n_actions))

        else:
            action =np.argmax(self.predict(state))
        
        self.oldAction = action
        return action


def play_one(model, ft,env, eps):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        state=ft.transform([observation])

        action = model.get_action(state, eps)
        #prev_observation = observation
        observation, reward, done, _ = env.step(action)
        model.learn(ft.transform([observation]),reward)
        totalreward += reward
        iters +=1

    return totalreward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict([_])), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
        rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main(show_plots=True):
    env = gym.make('MountainCar-v0').env
    ft = FeatureTransformer(env)
    model = Q_model("constant", env.action_space.n)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)


    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        # eps = 1.0/(0.1*n+1)
        eps = 0.1*(0.97**n)
        if n == 199:
            print("eps:", eps)
        # eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, ft,env, eps)
        totalrewards[n] = totalreward
        if (n + 1) % 100 == 0:
            print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    if show_plots:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(totalrewards)

        # plot the optimal state-value function
        plot_cost_to_go(env, model)


if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
    main()








        # update the model
        #next = model.predict(observation)
        # assert(next.shape == (1, env.action_space.n))
        # G = reward + gamma*np.max(next[0])
        # model.update(prev_observation, action, G)



#%%
