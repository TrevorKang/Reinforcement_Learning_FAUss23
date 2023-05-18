import numpy as np
from helper import action_value_plot, test_agent
from random import random
from gym_gridworld import GridWorldEnv


class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        # Get gridworld state space dimensionality
        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])

        # Get number of possible actions
        self.num_actions = env.action_space.n

        # TODO: define a Q-function member variable self.Q
        # Q[y, x, z] is value of action z for grid position y, x
        self.Q = np.zeros((self.grid_height, self.grid_height, self.num_actions))
        # set Q(terminal, _ ) to 0
        self.Q[3, 0] = 0
        self.Q[3, 1] = 0
        self.Q[3, 3] = 0

    def action(self, s, epsilon=None):

        # TODO: implement epsilon-greedy action selection
        action_prob = np.ones(self.num_actions, dtype=float) * epsilon / self.num_actions
        best_action = np.argmax(self.Q[s[0], s[1]])
        action_prob[best_action] += (1.0 - epsilon)
        # p = random()
        # if p < epsilon:
        #     return self.env.action_space.sample()  # returns random action
        # else:
        #     return best_action  # return the action which brings the max. reward
        action = np.random.choice(np.arange(self.num_actions), p=action_prob)
        return action


class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=20000):
        # 0 up
        # 2 down
        # 1 right
        # 3 left
        # TODO: implement training loop
        s, info = self.env.reset()
        a = self.action(s=s, epsilon=self.eps)
        for i in range(n_timesteps):
            s_, reward, terminal, _, _ = self.env.step(a)
            a_ = self.action(s=s_, epsilon=self.eps)
            self.update_Q(s=s, a=a_, r=reward, s_=s_, a_=a_)
            s = s_
            a = a_
            if terminal is True:
                s, _ = self.env.reset()

    def update_Q(self, s, a, r, s_, a_):

        # TODO: implement Q-value update rule
        current_q = self.Q[s[0], s[1], a]
        next_q = self.Q[s_[0], s_[1], a_]
        self.Q[s[0], s[1], a] = current_q + self.lr * (r + self.g * next_q - current_q)


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):

        # TODO: implement training loop
        s, info = self.env.reset()
        for i in range(n_timesteps):
            a = self.action(s, epsilon=self.eps)
            s_, r, terminal, _, _ = self.env.step(a)
            self.update_Q(s, a, r, s_)
            s = s_
            if terminal is True:
                s, _ = self.env.reset()

    def update_Q(self, s, a, r, s_):

        # TODO: implement Q-value update rule
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * \
                                (r + self.g * max(self.Q[s_[0], s_[1]]) - self.Q[s[0], s[1], a])


def run():
    # Create environment
    # env = GridWorldEnv(map_name='standard')
    env = GridWorldEnv(map_name='cliffwalking')

    # Hyperparameters
    discount_factor = 0.9
    learning_rate = 0.05
    epsilon = 0.4
    n_timesteps = 200000

    # Train SARSA agent
    # sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    # sarsa_agent.learn(n_timesteps=n_timesteps)
    # action_value_plot(sarsa_agent)
    # print('Testing SARSA agent')
    # test_agent(sarsa_agent, env, epsilon=0.1)

    # Train Q-Learning agent
    qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    qlearning_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(qlearning_agent)
    print('Testing Q-Learning agent')
    test_agent(qlearning_agent, env, epsilon=0.1)


if __name__ == "__main__":
    run()
