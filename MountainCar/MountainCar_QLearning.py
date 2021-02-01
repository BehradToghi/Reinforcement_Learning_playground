import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import json
import time
import argparse

class QLearning:
    def __init__(self, args):
        self.env = gym.make('MountainCar-v0')
        self.state = self.env.reset()
        self.episodes = args.episodes
        self.initial_epsilon = args.epsilon
        self.epsilon = args.epsilon
        self.resolution = args.resolution
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.alpha_decay = args.alpha_decay
        self.decay_steps = args.decay_steps
        self.discrete_state = self.convert_to_discrete_state(self.state)
        self.q_table = self.create_q_table()

        if args.alpha_decay == 2:
            self.visit_table = np.ones([self.env.action_space.n] + self.resolution)

    def create_q_table(self):
        _q_table = np.zeros([self.env.action_space.n] + self.resolution)
        return _q_table

    def convert_to_discrete_state(self, state):
        _delta = (self.env.observation_space.high - self.env.observation_space.low) / self.resolution
        _discrete_state = np.round((state - self.env.observation_space.low) / _delta)
        _discrete_state = _discrete_state.astype(int)

        return _discrete_state

    def get_action(self):
        if (np.random.rand() <= self.epsilon):
            #explore
            action = self.env.action_space.sample()
        else:
            #exploit
            _q_sub_table = self.q_table[:, self.discrete_state[0], self.discrete_state[1]]
            action = np.argmax(_q_sub_table)

        return action

    def get_learning_rate(self, episode=0, visits=0):
        if self.alpha_decay == 0:
            pass
        if self.alpha_decay == 1:
            self.alpha = self.alpha / (self.alpha + episode/(self.episodes/self.decay_steps))
        if self.alpha_decay == 2:
            self.alpha = 1 / visits
            print("visits, self.alpha = ", visits, self.alpha)

    def decay_epsilon(self):
        _step = (self.initial_epsilon - 0.01) / (self.episodes * 3/4)
        if self.epsilon >= 0.01:
            self.epsilon -= _step

    def test(self, path):
        with open (path, 'rb') as file:
            self.q_table = pickle.load(file)

        self.env.reset()
        self.discrete_state = self.convert_to_discrete_state(self.state)
        done = False

        while not done:
            self.env.render()
            _action = np.argmax(self.q_table[:, self.discrete_state[0], self.discrete_state[1]])
            _next_state, reward, done, info = self.env.step(_action)
            _next_state = self.convert_to_discrete_state(_next_state)
            _next_state = _next_state.astype(int)
            self.discrete_state = _next_state

        time.sleep(5)
        self.env.close()
    
    def train(self):
        print("INFO: Learning for {0:d} episodes.".format(self.episodes))
        _rewards = [-200]
        _episode_reward = []

        for episode in range(self.episodes):
            self.env.reset()
            self.discrete_state = self.convert_to_discrete_state(self.state)
            done = False
            _episode_reward = []

            if self.alpha_decay == 0:
                self.get_learning_rate()
            if self.alpha_decay == 1:
                self.get_learning_rate(episode=episode)

            while not done:
                _action = self.get_action()
                _next_state, reward, done, info = self.env.step(_action)
                _next_state = self.convert_to_discrete_state(_next_state)
                _next_state = _next_state.astype(int)

                # print("INFO: EPISODE # {:d} ------ episode_reward, next_state, reward, action, state, done, info,"
                #       " epsilon = " .format(episode), _rewards[-1], _next_state, reward, _action, self.discrete_state,
                #       done, info, self.epsilon)

                if self.alpha_decay == 2:
                    self.visit_table[_action, self.discrete_state[0], self.discrete_state[1]] += 1
                    visits = self.visit_table[_action, self.discrete_state[0], self.discrete_state[1]]
                    self.get_learning_rate(visits=visits)

                target = reward + self.gamma * np.max(self.q_table[:, _next_state[0], _next_state[1]])
                _episode_reward.append(reward)

                if done and not bool(info):
                    self.q_table[_action, self.discrete_state[0], self.discrete_state[1]] = sum(_episode_reward)
                    print("INFO: car reached the goal after {:f} steps".format(sum(_episode_reward)))
                else:
                    self.q_table[_action, self.discrete_state[0], self.discrete_state[1]] = \
                        self.q_table[_action, self.discrete_state[0], self.discrete_state[1]] \
                        + self.alpha * (target - self.q_table[_action, self.discrete_state[0], self.discrete_state[1]])
                self.discrete_state = _next_state

            self.decay_epsilon()
            _rewards.append(sum(_episode_reward))

            print("INFO: EPISODE #{:d} ------ episode_reward = {:0.1f}, epsilon = {:0.3f}, alpha = {:0.3f}"
                  .format(episode, _rewards[-1], self.epsilon, self.alpha))
        self.env.close()

        return _rewards, self.q_table

def plot_rewards(values, scenario_name):
    _fig, _ax = plt.subplots()
    _ax.plot(np.linspace(1, len(values), len(values)), values)
    _ax.set(xlabel='Episode', ylabel='Average Episode Reward',
           title='MountainCar Q-Learning')
    _ax.grid()
    plt.xlim(0, len(values))
    _path = os.path.join(".", "output_"+scenario_name, "rewards_plot_" + scenario_name + ".png")
    plt.savefig(_path)
    plt.show()

def save_results(q_table, configs, scenario_name):
    _path = os.path.join(".", "output_"+scenario_name)
    if not os.path.exists(_path):
        os.mkdir(_path)
    with open (os.path.join(_path, "q_table_"+scenario_name+".pkl"), 'wb') as file:
        pickle.dump(q_table, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(_path, "configs_" + scenario_name + ".json"), 'w') as file:
        json.dump(configs, file)

def main(args):
    agent = QLearning(args)

    if (args.mode == "train") or (args.mode == "both"):
        print(">>>>>>>>> INFO: Training Experiment with this configs: ", vars(args))
        _rewards, _q_table = agent.train()
        save_results(_q_table, vars(args), args.scenario)
        plot_rewards (_rewards, args.scenario)
    elif (args.mode == "test") or (args.mode == "both"):
        print(">>>>>>>>> INFO: Testing Experiment with this configs: ", vars(args))
        if args.checkpoint_path is None:
            _checkpoint_path = os.path.join(".", "output_"+args.scenario, "q_table_"+args.scenario+".pkl")
        else:
            _checkpoint_path = args.checkpoint_path
        agent.test(_checkpoint_path)

if __name__ == '__main__':
    # Reading input arguments from command line
    parser = argparse.ArgumentParser(description='Simple Q-learning for OpenAI Gym MountainCar-v0 environment.')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of Training Episodes.')
    parser.add_argument('--epsilon', type=float, default=0.9, help='Exploration probability (e-greedy).')
    parser.add_argument('--resolution', type=list, default=[20, 20], help='Resolution to Discretize the State-space.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning Rate.')
    parser.add_argument('--alpha_decay', type=int, default=0,
                        help='Learning Rate Decay: 0=no decay, 1=exponential decay, 2=visit counter decay.')
    parser.add_argument('--decay_steps', type=int, default=3,
                        help='Number of Learning Rate Decay Steps (only works with --alpha_decay=1).')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount Factor.')
    parser.add_argument('--mode', type=str, default="both", help='train/test/both.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoints. If not provided, will try to match with scenario name.')
    parser.add_argument('--scenario', type=str, default="experiment", help='Name of the Scenario.')
    args = parser.parse_args()

    # Run the training
    main(args)
