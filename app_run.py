from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

from tqdm import tqdm

from collections import deque

from environment import Environment

import numpy as np
import random
import time
import os

LOAD_MODEL = "models/2x256___874.00max__440.44avg_-176.00min__1617462987.model"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

class DQNAgent:

    def __init__(self, env):
        # Main model (gets trained every step)
        self.model = self.create_model(env)

        # Target model (this is what we .predict against every step)
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

        self.env = env

    def create_model(self, env):

        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")

        else:
            model = Sequential()

            model.add(Conv2D(256, (3, 3), input_shape=np.zeros((15, 15, 3)).shape))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.get_action_space_size(), activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        # Feature sets (images from game)
        X = []

        # Labels (action we decide to take)
        y = []

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def main():

    # Exploration settings
    epsilon = 0  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    print("Running Inference App...")

    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    #random.seed(1)
    #np.random.seed(1)
    #tf.set_random_seed(1)

    # Create models folder
    if not os.path.isdir("models"):
        os.mkdir("models")

    # Create Environment
    env = Environment("data/map_small_edge.jpg", 15, 100, 100, 200)

    agent = DQNAgent(env)

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):

        episode_reward = 0
        step = 1
        current_state = env.reset()

        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state, step))
            else:
                action = np.random.randint(0, env.get_action_space_size())

            new_state, reward, done = env.step(action)
            episode_reward += reward

            if SHOW_PREVIEW and not episode % 1:
                env.render_map()
                env.render_sub_map()

            #agent.update_replay_memory((current_state, action, reward, new_state, done))
            #agent.train(done, step)

            current_state = new_state
            step += 1

        print("reward: {}".format(reward),
              "total reward: {}".format(env.agent_reward),
              "game_over: {}".format(done),
              "total_gen_reward: {}".format(env.total_generated_rewards)
              )


if __name__ == '__main__':
    main()
