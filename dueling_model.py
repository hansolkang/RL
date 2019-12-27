import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    replay_memory = 10000
    batch_size = 32
    discount_factor = 0.99
    len_input = 4

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None

        # last of len_input is (present+past+past...)
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.len_input])
        self.input_A = tf.placeholder(tf.int64, [None]) # action
        self.input_Y = tf.placeholder(tf.float32,[None]) # loss

        self.Q = self.build_network('main')
        self.target_Q = self.build_network('target')
        self.cost, self.train_op = self.build_op()

    def build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            value = tf.layers.dense(model, 1, activation=tf.nn.relu)
            advantage = tf.layers.dense(model, self.n_action, activation=tf.nn.relu)
            output = value + (advantage - tf.reduce_mean(advantage, axis = 1, keep_dims=True))
            Q = tf.layers.dense(output, self.n_action, activation=None)
        return output

    def build_op(self):

        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    def init_state(self,state):
        state = [state for _ in range(self.len_input)]
        self.state = np.stack(state,axis=2)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:,:,1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.replay_memory:
            self.memory.popleft()

        self.state = next_state
    def sample_memory(self):
        sample_memory = random.sample(self.memory, self.batch_size)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        state, next_state, action, reward, terminal = self.sample_memory()

        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X:next_state})
        # Q_value = self.session.run(self.Q, feed_dict={self.input_X: next_state})

        # if episode is terminates at step j+1 then r_j


        Y = []
        for i in range(self.batch_size):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.discount_factor * np.max(target_Q_value[i])) # nature 2015
                # Y.append(reward[i] + self.discount_factor * np.max(Q_value[i])) # nature 2013
                # a = np.argmax(Q_value[i])
                # Y.append(reward[i] + self.discount_factor * (target_Q_value[i][a]))  # double dqn

        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })