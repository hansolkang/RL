"""
DQN (NIPS 2013)

Playing Atari with Deep Reinforcement Learning
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque
from gym.envs.registration import register
import gym
# Register CartPole with user-defined max_episode_steps
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold=10000.0,
)

from gym import wrappers
env = gym.make('CartPole-v2')
# env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 64

# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0,INPUT_SIZE)
    y_stack = np.empty(0).reshape(0,OUTPUT_SIZE)
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        # if terminal
        if done :
            Q[0,action] = reward
        # if not terminal
        else :
            Q[0,action] = reward + DISCOUNT_RATE * np.max(DQN.predict(next_state))
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])
    return DQN.update(x_stack,y_stack)

def bot_play(mainDQN):
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    # store the previous observations in replay memory
    replay_buffer = deque()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        for episode in range(MAX_EPISODE):
            e = 1. / ((episode / 10) + 1)
            done = False
            state = env.reset()
            step_count = 0

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)
                # give big panelty
                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                step_count += 1

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

            print("Episode : {}    steps: {}".format(episode, step_count))
            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer,10)
                    loss, _ = simple_replay_train(mainDQN,minibatch)
                print("Loss : ", loss)

                # if len(replay_buffer) > BATCH_SIZE:
                #     minibatch = random.sample(replay_buffer, BATCH_SIZE)
                #     simple_replay_train(mainDQN, minibatch)

                bot_play(mainDQN)

if __name__ == "__main__":
    main()
