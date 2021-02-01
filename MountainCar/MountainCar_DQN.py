import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
import json
import time
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import tensorflow as tf
import keras
# from MountainCar import MountainCarEnv
from datetime import datetime
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import argparse
torch.manual_seed(1); np.random.seed(1)

class DQN:
    # def __init__(self, episodes=1000, epsilon=0.9, resolution=[20, 20], alpha=0.2, gamma=0.9, min_eps=0,q_table_ini="zero",decay= 'linear'):
    def __init__(self, episodes=1000, iteration_num=201, learning_rate=0.001, gamma=0.9, epsilon=0.9, epsilon_min=0.01, epsilon_decay_type='fix', epsilon_decay=0.05, replay_buffer_size=2000, gpu=False, average_range=100, pytorch=False, original_reward=True):
        # Import and initialize Mountain Car Environment
        self.env = gym.make('MountainCar-v0')
        # self.env.seed(1)
        # self.env= MountainCarEnv()
        # self.state = self.env.reset()
        self.episodes_num = episodes
        self.gpu=gpu
        self.original_reward=original_reward

        self.learning_rate = learning_rate #alpha  # learingRate
        self.gamma = gamma

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay_type = epsilon_decay_type
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # self.epsilon_decay = (self.epsilon - self.min_eps) / self.episodes

        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.pytorch=pytorch
        if self.pytorch:
            self.hidden = 100
            self.loss_fn= None
            self.optimizer=None
            self.scheduler=None
            self.train_network = self.create_network_pytorch()
            self.use_target_network = False
        else:
            self.train_network = self.create_network()
            self.use_target_network = True
            self.target_network = self.create_network()
            self.target_network.set_weights(self.target_network.get_weights())




        self.iteration_num = iteration_num  # max is 200
        self.num_pick_from_buffer = 32




        self.save_sucess_model = False

        self._rewards_by_episode = []
        self._loss_by_episode = []
        self._position_by_episode = []

        self.average_range = average_range
        self.path=" "
        if gpu:
            config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} )
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)

        self.writer = SummaryWriter('~/tboardlogs/{}'.format(datetime.now().strftime('%b%d_%H-%M-%S')))
        self.configs = self.create_log()

    def create_log(self):
        _configs = {"episodes": self.episodes_num,
                    "iterationNum":self.iteration_num,
                    "learingRate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_min":self.epsilon_min,
                    "epsilon_decay_type":self.epsilon_decay_type,
                    "epsilon_decay":self.epsilon_decay,
                    # "replayBuffer_size":self.replayBuffer_size,
                    "gpu":self.gpu,
                    "average_range":self.average_range,
                    "pytorch":self.pytorch

                    }

        return _configs

    def get_action(self,state):
        # Determine next action - epsilon greedy strategy
        if (np.random.rand(1) < self.epsilon):
            # explore
            # action = self.env.action_space.sample()
            action = np.random.randint(0, self.env.action_space.n)
            # action = np.random.randint(0, 3)
        else:
            # exploit
            if self.pytorch:
                Q = self.train_network(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
                _, action = torch.max(Q, -1)
                action = action.item()
            else:
                action = np.argmax(self.train_network.predict(state.reshape(1, 2))[0])

        return action

    def get_learning_rate(self):
        # TODO: decay learning rate
        return self.learning_rate

    def decay_epsilon(self):


        if self.epsilon_decay_type== 'fix':

            # reduction = (self.epsilon - self.min_eps) / self.episodes
            """
            If we assume an epsilon-greedy exploration strategy where epsilon 
            decays linearly to a specified minimum (min_eps) over the total number of episodes, 
            """
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        elif self.epsilon_decay_type=='exponential':
            # Adjust epsilon
            self.epsilon *= self.epsilon_decay

        else:
            step = (self.initial_epsilon - 0.01) / (self.episodes_num * 3 / 4)
            if self.epsilon >= 0.01:
                self.epsilon -= step

        self.epsilon = max(self.epsilon_min, self.epsilon)

    def show_info(self):
        print('State space: ', self.env.observation_space)
        print('Action space: ', self.env.action_space)
        print('State space low: ',self.env.observation_space.low)
        print('State space high: ',self.env.observation_space.high)

    def random_actions(self):
        for _ in range(1000):
            self.env.render()
            state, reward, done, info = self.env.step(self.env.action_space.sample())
            print("DEBUG: state, reward, done, info = ", state, reward, done, info)

    def hard_code_policy(self):
        # follow a hard-coded policy
        for _ in range(50):
            self.env.render()
            for left in range(34):
                state, reward, done, info = self.env.step(0)
                print("DEBUG: state, reward, done, info = ", state, reward, done, info)
            for right in range(34):
                state, reward, done, info = self.env.step(2)
                print("DEBUG: state, reward, done, info = ", state, reward, done, info)


    def create_network_pytorch(self):
        torch.manual_seed(1)
        model=torch.nn.Sequential(
            torch.nn.Linear(self.env.observation_space.shape[0], self.hidden, bias=False),
            torch.nn.Linear(self.hidden, self.env.action_space.n, bias=False),

        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        return model


    def create_network(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n,activation='linear'))
        # model.compile(optimizer=optimizers.RMSprop(lr=self.learingRate), loss=losses.mean_squared_error)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def create_network_1layer(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(layers.Dense(50, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        # model.compile(optimizer=optimizers.RMSprop(lr=self.learingRate), loss=losses.mean_squared_error)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # def createNetwork3layer(self):
    #     model = models.Sequential()
    #     state_shape = self.env.observation_space.shape
    #
    #     model.add(layers.Dense(128, activation='relu', input_shape=state_shape))
    #     model.add(layers.Dense(64, activation='relu'))
    #     model.add(layers.Dense(32, activation='relu'))
    #     model.add(layers.Dense(self.env.action_space.n, activation='linear'))
    #     # model.compile(optimizer=optimizers.RMSprop(lr=self.learingRate), loss=losses.mean_squared_error)
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     return model

    def train_from_buffer_boost(self):
        if len(self.replay_buffer) < self.num_pick_from_buffer:
            return
        samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)
        npsamples = np.array(samples)
        states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
        rewards = rewards_temp.reshape(self.num_pick_from_buffer, ).astype(float)
        targets = self.train_network.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.target_network.predict(newstates).max(axis = 1)
        targets[(np.arange(self.num_pick_from_buffer), actions_temp.reshape(self.num_pick_from_buffer, ).astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma) * notdones
        self.train_network.fit(states, targets, epochs=1, verbose=0)

    def train_from_buffer(self):
        if len(self.replay_buffer) < self.num_pick_from_buffer:
            return

        samples = random.sample(self.replay_buffer, self.num_pick_from_buffer)

        states = []
        newStates=[]
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)

        newArray = np.array(states)
        states = newArray.reshape(self.num_pick_from_buffer, 2)

        new_array2 = np.array(newStates)
        newStates = new_array2.reshape(self.num_pick_from_buffer, 2)

        targets = self.train_network.predict(states)
        new_state_targets=self.target_network.predict(newStates)

        i=0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i+=1

        self.train_network.fit(states, targets, epochs=1, verbose=0)

    def train_pytorch(self,state,action,reward,next_state,done):
        # Find max Q for t+1 state
        q=self.train_network(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
        q_next=self.train_network(Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
        max_q, _ = torch.max(q_next, -1)
        # Create target Q value for training the policy
        q_target = q.clone()
        q_target = Variable(q_target.data)
        q_target[action] = reward + torch.mul(max_q.detach(), self.gamma)

        # Calculate loss
        loss = self.loss_fn(q, q_target)

        # Update policy
        self.train_network.zero_grad()
        loss.backward()
        self.optimizer.step()


        if done:
            if next_state[0] >= 0.5:
                # Adjust learning rate
                self.scheduler.step()
        return loss.item()


    def test(self, path):
        # with open(path, 'rb') as file:
        #     self.q_network = pickle.load(file)
        if self.pytorch:
            model=self.create_network_pytorch()
            model.load_state_dict(torch.load(path))
        else:
            model = models.load_model(path)

        self.env.reset()
        for i_episode in range(5):
            current_state = self.env.reset()

            print("============================================")

            reward_sum = 0
            for t in range(200):
                self.env.render()
                if self.pytorch:
                    _, action = torch.max(model(Variable(torch.from_numpy(current_state).type(torch.FloatTensor))),
                                          -1)
                    action = action.item()

                else:
                    action = np.argmax(model.predict(current_state.reshape(1, 2))[0])

                new_state, reward, done, info = self.env.step(action)

                new_state = new_state

                current_state = new_state

                reward_sum += reward
                if done:
                    print("Episode finished after {} timesteps reward is {}".format(t + 1, reward_sum))
                    break

        time.sleep(5)
        self.env.close()

    def train(self):
        print("INFO: Learning for {0:d} episodes.".format(self.episodes_num))
        # _rewards_by_episode = []
        # episode_reward = []
        # _position_by_episode = []
        ave_reward_list=[]
        reward_list = []
        successes=0
        self.env.seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        current_state = self.env.reset()

        # Run Q learning algorithm
        # for episode in range(self.episodesNum):
        for episode in trange(self.episodes_num):
            # self.env.reset()
            current_state = self.env.reset()

            done = False
            episode_reward = []
            episode_loss =[]
            max_position = -99
            # max_position = -0.4

            for i in range(self.iteration_num):
            # while not done:
                # Render environment for last five episodes or show the animation every 50 eps
                if episode >= (self.episodes_num - 5) or episode % 50 == 0:
                    self.env.render()


                _action = self.get_action(current_state)
                _next_state, reward, done, info = self.env.step(_action)


                # # Keep track of max position
                if _next_state[0] > max_position:
                    max_position = _next_state[0]

                # print("INFO: EPISODE # {:d} ------ episode_reward, next_state, reward, action, state, done, info,"
                #       " epsilon = " .format(episode), _rewards[-1], _next_state, reward, _action, self.discrete_state,
                #       done, info, self.epsilon)

                if self.pytorch ==False:
                    if _next_state[0] >= 0.5:
                        reward += 10
                else:
                    if not self.original_reward:
                        # Adjust reward based on car position
                        reward = _next_state[0] + 0.5

                        # # Adjust reward for task completion
                        if _next_state[0] >= 0.5:
                            reward += 1

                alpha = self.get_learning_rate()
                episode_reward.append(reward)



                if self.pytorch:
                    #update with just replay buffer
                    self.replay_buffer.append([current_state, _action, reward, _next_state, done])
                    loss=self.train_pytorch(current_state,_action,reward,_next_state,done)
                    episode_loss.append(loss)
                else:
                    # Or you can use self.trainFromBuffer_Boost(), it is a matrix wise version for boosting
                    self.replay_buffer.append([current_state.reshape(1, 2), _action, reward, _next_state.reshape(1, 2), done])
                    if self.gpu:
                        self.train_from_buffer_boost()
                    else:
                        self.train_from_buffer()

                # currentState = _next_state

                if done:
                    if _next_state[0] >= 0.5:
                        # On successful epsisodes, adjust the following parameters

                        # Adjust epsilon
                        # epsilon *= .99

                        # Record successful episode
                        successes += 1

                        # Adjust epsilon
                        if self.epsilon_decay_type=="exponential":
                            self.decay_epsilon()
                        # self.epsilon *= .95

                        self.writer.add_scalar('data/epsilon', self.epsilon, episode)

                        # self.writer.add_scalar('data/learning_rate', self.optimizer.param_groups[0]['lr'], episode)

                        self.writer.add_scalar('data/cumulative_success', successes, episode)
                        self.writer.add_scalar('data/success', 1, episode)
                    else:
                        self.writer.add_scalar('data/success', 0, episode)

                    self.writer.add_scalar('data/episode_loss', sum(episode_loss), episode)
                    self.writer.add_scalar('data/episode_reward', sum(episode_reward), episode)
                    # weights = np.sum(np.abs(policy.l2.weight.data.numpy())) + np.sum(np.abs(policy.l1.weight.data.numpy()))
                    # writer.add_scalar('data/weights', weights, episode)
                    self.writer.add_scalar('data/position', _next_state[0], episode)

                    self._position_by_episode.append(_next_state[0])
                    self._rewards_by_episode.append(sum(episode_reward))
                    self._loss_by_episode.append(sum(episode_loss))
                    break
                else:
                    current_state = _next_state

            # if episode %50 ==0:
            #     print("INFO: EPISODE #{:d} ------  episode_reward = {:0.1f}, epsilon = {:0.3f}"
            #           .format(episode + 1, sum(episode_reward), self.epsilon))

            reward_list.append(sum(episode_reward))

            if (episode + 1) % self.average_range == 0:
                ave_reward_list.append(np.mean(reward_list))
                reward_list = []
                print("INFO: EPISODE #{:d} ------ average_reward = {:0.1f}, episode_reward = {:0.1f}, epsilon = {:0.3f}"
                  .format(episode+1,ave_reward_list[-1], self._rewards_by_episode[-1], self.epsilon))

            if self.save_sucess_model:
                if i >= self.iteration_num:
                    print("Failed to finish task in epsoide {}".format(episode))
                else:
                    print("Success in epsoide {}, used {} iterations!".format(episode, i))
                    self.train_network.save('./trainNetworkInEPS{}.h5'.format(episode))

            if self.epsilon_decay_type == "fix":
                self.decay_epsilon()

            if self.use_target_network:
                # Sync
                self.target_network.set_weights(self.train_network.get_weights())

        # if i >= 199:
        #     print("Failed to finish task in epsoide {}".format(episode))
        # else:
        #     print("Success in epsoide {}, used {} iterations!".format(episode, i))
        #     self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(episode))

        self.writer.close()
        self.env.close()
        print('INFO: successful episodes: {:d} - {:.4f}%'.format(successes, successes / self.episodes_num * 100))

        return self._rewards_by_episode, self.train_network, self.configs

    def plot_rewards(self, scenario_name):
        rewards = []
        average_range = self.average_range
        for i in range(0, len(self._rewards_by_episode), average_range):
            if i + average_range <= self.episodes_num:
                rewards.append(np.mean(self._rewards_by_episode[i:i + average_range]))
            else:
                break
                # rewards.append(np.mean(self._rewards_by_episode[i:]))

        plt.plot(average_range * (np.arange(len(rewards)) + 1), rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes, MountainCar Q-Learning')
        _path = os.path.join(self.path, "rewards_plot_" + scenario_name + ".png")
        # plt.savefig('MountainCar_QLearning_rewards.jpg')
        plt.savefig(_path)
        plt.show()
        plt.close()

    def plot_position(self,scenario_name):
        # Plot Results
        # Around episode 1000 the agent begins to successfully complete episodes.
        plt.figure(2, figsize=[10, 5])
        p = pd.Series(self._position_by_episode)
        ma = p.rolling(10).mean()
        plt.plot(p, alpha=0.8)
        plt.plot(ma)
        plt.xlabel('Episode')
        plt.ylabel('Position')
        plt.title('Car Final Position')
        _path = os.path.join(self.path, "final_position_" + scenario_name + ".png")
        plt.savefig(_path)

    def plot_policy(self, scenario_name):

        # Visualize Policy¶
        # We can see the policy by plotting the agent’s choice over a combination of positions and velocities. You can see that the agent learns to, usually, move left when the car’s velocity is negative and then switch directions when the car’s velocity becomes positive with a few position and velocity combinations on the left side of the environment where the agent will do nothing.
        X = np.random.uniform(-1.2, 0.6, 10000)
        Y = np.random.uniform(-0.07, 0.07, 10000)
        Z = []

        for i in range(len(X)):

            if self.pytorch:
                _, temp = torch.max(self.train_network(Variable(torch.from_numpy(np.array([X[i], Y[i]]))).type(torch.FloatTensor)),
                                    dim=-1)
                z = temp.item()
            else:
                # p=self.trainNetwork.predict(np.array([X[i], Y[i]]).reshape(1,2))
                z=np.argmax(self.train_network.predict(np.array([X[i], Y[i]]).reshape(1, 2))[0])

            Z.append(z)

        Z = pd.Series(Z)
        colors = {0: 'blue', 1: 'lime', 2: 'red'}
        colors = Z.apply(lambda x: colors[x])
        labels = ['Left', 'Right', 'Nothing']
        fig = plt.figure(3, figsize=[7, 7])
        ax = fig.gca()
        plt.set_cmap('brg')
        surf = ax.scatter(X, Y, c=Z)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Policy')
        recs = []
        for i in range(0, 3):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=sorted(colors.unique())[i]))
        plt.legend(recs, labels, loc=4, ncol=3)

        _path = os.path.join(self.path, "policy_" + scenario_name + ".png")
        fig.savefig(_path)
        plt.show()

    def save_results(self,scenario_name):
        # _path = os.path.join(".", "output", scenario_name + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        folder=_path = os.path.join(".", "output")
        if not os.path.exists(folder):
            os.mkdir(folder)
        _path = os.path.join(".", "output", scenario_name + datetime.now().strftime('-%Y-%m-%d_%H-%M'))
        self.path=_path
        if not os.path.exists(_path):
            os.mkdir(_path)

        self.plot_rewards(scenario_name)
        self.plot_policy(scenario_name)
        self.plot_position(scenario_name)
        # with open(os.path.join(_path, "q_network_" + scenario_name + ".pkl"), 'wb') as file:
        #     pickle.dump(self.trainNetwork, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(_path, "data_" + scenario_name + ".pkl"), 'wb') as file:
            data={"_rewards_by_episode":self._rewards_by_episode,
                  "_loss_by_episode":self._loss_by_episode,
                  "_position_by_episode":self._position_by_episode
                  }
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(_path, "configs_" + scenario_name + ".json"), 'w') as file:
            json.dump(self.configs, file)
        # with open(os.path.join(_path, "agent_" + scenario_name + ".pkl"), 'wb') as file:
        #     pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

        if self.pytorch:
            torch.save(self.train_network.state_dict(), os.path.join(_path, 'trainNetworkInEPS{}last.pth'.format(self.episodes_num)))
        else:
            self.train_network.save(os.path.join(_path, 'trainNetworkInEPS{}last.h5'.format(self.episodes_num)))

def plot_policy(agent, path, scenario_name):
        if agent.pytorch:
            model = agent.create_network_pytorch()
            model.load_state_dict(torch.load(path))
        else:
            model = models.load_model(path)
        # Visualize Policy¶
        # We can see the policy by plotting the agent’s choice over a combination of positions and velocities. You can see that the agent learns to, usually, move left when the car’s velocity is negative and then switch directions when the car’s velocity becomes positive with a few position and velocity combinations on the left side of the environment where the agent will do nothing.
        X = np.random.uniform(-1.2, 0.6, 10000)
        Y = np.random.uniform(-0.07, 0.07, 10000)
        Z = []

        for i in range(len(X)):

            if agent.pytorch:
                _, temp = torch.max(model(Variable(torch.from_numpy(np.array([X[i], Y[i]]))).type(torch.FloatTensor)),
                                    dim=-1)
                z = temp.item()
            else:
                # p=self.trainNetwork.predict(np.array([X[i], Y[i]]).reshape(1,2))
                z=np.argmax(model.predict(np.array([X[i], Y[i]]).reshape(1,2))[0])

            Z.append(z)

        Z = pd.Series(Z)
        colors = {0: 'blue', 1: 'lime', 2: 'red'}
        colors = Z.apply(lambda x: colors[x])
        labels = ['Left', 'Right', 'Nothing']
        fig = plt.figure(3, figsize=[7, 7])
        ax = fig.gca()
        plt.set_cmap('brg')
        surf = ax.scatter(X, Y, c=Z)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Policy')
        recs = []
        for i in range(0, 3):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=sorted(colors.unique())[i]))
        plt.legend(recs, labels, loc=4, ncol=3)

        _path = os.path.join("policy_" + scenario_name + ".png")
        fig.savefig(_path)
        plt.show()

def main(args):
    #for debug
    arg_parse = True

    # def __init__(self, episodes=1000, iteration_num=201, learning_rate=0.001, gamma=0.9, epsilon=0.9, epsilon_min=0.01, epsilon_decay_type='fix', epsilon_decay=0.05, replay_buffer_size=2000, gpu=False, average_range=100, pytorch=False, original_reward=True):

    if arg_parse== True:
        agent = DQN(episodes=args.episodes, iteration_num=args.iteration_num, learning_rate=args.learning_rate, gamma=args.gamma, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay_type=args.epsilon_decay_type, epsilon_decay=args.epsilon_decay, replay_buffer_size=args.replay_buffer_size, gpu=args.gpu, average_range=args.average_range, pytorch=args.pytorch, original_reward=args.original_reward)
        scenario_name=args.scenario_name
        if args.mode=='train':
             _rewards_by_episode, qnetwork, configs = agent.train()
             agent.save_results(scenario_name)

        if args.mode=='test':
            agent.path =args.checkpoint_path
            agent.test(os.path.join(agent.path, 'trainNetworkInEPS{}last.pth'.format(agent.episodes_num)))

        if args.mode == 'both':
            _rewards_by_episode, qnetwork, configs = agent.train()
            agent.save_results(scenario_name)
            agent.test(os.path.join(agent.path, 'trainNetworkInEPS{}last.pth'.format(agent.episodes_num)))


    else:
        scenario_name = "MountainCar_DQN"
        training= False  # Trainin True , create and train the model, training False just test a previous trained model.
        model='keras'   # Model = keras , DQN with keras model; model = pytorch , DQN with pytorch model and original reward,  model = pytorch_new_reward , DQN with pytorch model and a modified reward,
        # Train examples

        if model == 'keras':
            agent = DQN(episodes=400, iteration_num=201, learning_rate=0.001, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decay_type='fix', epsilon_decay=0.05, replay_buffer_size=20000, gpu=False, average_range=100, pytorch=False, original_reward=False)
        elif model=='pytorch':
            agent = DQN(episodes=3000, epsilon=0.3, gamma=0.99, learning_rate = 0.001, iteration_num=2000, pytorch=True, epsilon_decay=0.99, epsilon_decay_type='exponential')
        elif model=='pytorch_new_reward':
            agent = DQN(episodes=1000, epsilon=0.3, gamma=0.99, learning_rate = 0.001, iteration_num=200, pytorch=True, epsilon_decay=0.95, epsilon_decay_type='exponential', original_reward=False)

        # agent.random_actions()
        # agent.hard_code_policy()

        if training:
            _rewards_by_episode,qnetwork, configs = agent.train()
            agent.save_results(scenario_name)

        # agent.path = "./output/MountainCar_DQN-2020-10-08_20-12_keras"
        # path = os.path.join(agent.path, 'trainNetworkInEPS{}last.h5'.format(agent.episodes_num))
        # plot_policy(agent, path, scenario_name)

        # load previus trainned models
        if training ==False:
            if agent.pytorch:
                if model=='pytorch':
                    agent.path = './output/MountainCar_DQN-2020-10-09_23-00_pytoch'
                    agent.test(os.path.join(agent.path,'trainNetworkInEPS{}last.pth'.format(agent.episodes_num)))
                elif model=='pytorch_new_reward':
                    agent.path = './output/MountainCar_DQN-2020-10-09_22-52_pytorch_modif_reward'
                    agent.test(os.path.join(agent.path, 'trainNetworkInEPS{}last.pth'.format(agent.episodes_num)))
            else:
                # agent.path = "./output/MountainCar_DQN-2020-10-08_20-12_keras"
                agent.path = "./output/MountainCar_DQN-2020-10-15_00-17"
                agent.test(os.path.join(agent.path,'trainNetworkInEPS{}last.h5'.format(agent.episodes_num)))
        # agent.test(os.path.join(".", "output_" + scenario_name, "q_network_" + scenario_name + ".pkl"))



if __name__ == '__main__':
    # Reading input arguments from command line
    parser = argparse.ArgumentParser(description='Simple Q-learning for OpenAI Gym MountainCar-v0 environment.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of Training Episodes.')
    parser.add_argument('--iteration_num', type=int, default=200, help='Number of Iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor.')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Exploration probability (e-greedy).')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help=' Minimum value for epsilon.')
    parser.add_argument('--epsilon_decay_type', type=str, default='exponential',help ='exponential/fix')
    parser.add_argument('--epsilon_decay', type=float, default=0.95, help=' Decay for fix linear decay.')
    parser.add_argument('--replay_buffer_size', type=int, default=2000, help=' Replay experience buffer size.')
    parser.add_argument('--gpu', type=bool, default=False, help=' Use or not GPU.')
    parser.add_argument('--average_range', type=int, default=100, help=' Average range for plotting rewards.')
    parser.add_argument('--pytorch', type=bool, default=True, help=' Use pytorch model.')
    parser.add_argument('--original_reward', type=bool, default=False, help=' Use original reward.')


    parser.add_argument('--mode', type=str, default="both", help='train/test/both.')
    parser.add_argument('--checkpoint_path', type=str, default="./output/MountainCar_DQN-2020-10-09_22-52_pytorch_modif_reward", help='path to model')
    parser.add_argument('--scenario_name', type=str, default="MountainCar_DQN", help='Name of the Scenario.')
    # parser.add_argument('--alpha_decay', type=int, default=0,
    #                     help='Learning Rate Decay: 0=no decay, 1=exponential decay, 2=visit counter decay.')
    # parser.add_argument('--resolution', type=list, default=[20, 20], help='Resolution to Discretize the State-space.')
    #
    # parser.add_argument('--decay_steps', type=int, default=3,
    #                     help='Number of Learning Rate Decay Steps (only works with --alpha_decay=1).')
    # parser.add_argument('--mode', type=str, default="both", help='train/test/both.')
    # parser.add_argument('--checkpoint_path', type=str, default=None,
    #                     help='Path to checkpoints. If not provided, will try to match with scenario name.')

    args = parser.parse_args()
    main(args)