from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

from tqdm import tqdm

from collections import deque

from gym.bug_env.environment import Environment

import numpy as np
import random
import time
import os

LOAD_MODEL = None

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 3  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20000

#  Stats settings
AGGREGATE_STATS_EVERY = 25  # episodes


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Override, to stop creating default log writer
    def set_model(self, model):
        pass

    # Override, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Override
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Override, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:

    learning_rate = 0.00001

    def __init__(self, env):
        # Main model (gets trained every step)
        self.model = self.create_model(env)

        # Target model (this is what we .predict against every step)
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

        self.env = env

    def create_model(self, env):

        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)
            print("Loaded previous model!")

        else:
            model = Sequential()

            model.add(Conv2D(256, (3, 3), input_shape=np.zeros((15, 15, 3)).shape))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))
      
            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.get_action_space_size(), activation="relu"))
            model.compile(loss="mse", optimizer=Adam(lr=DQNAgent.learning_rate), metrics=['accuracy'])

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

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def main():

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(tf.test.is_gpu_available())

    # Exploration settings
    epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99985
    MIN_EPSILON = 0.01

    print("Running Training App...")

    # For stats
    ep_rewards = [0]

    # For more repetitive results
    # random.seed(1)
    # np.random.seed(1)
    # tf.set_random_seed(1)

    # Create models folder
    if not os.path.isdir("models"):
        os.mkdir("models")

    # Create Environment
    env = Environment(map_img_path="gym/bug_env/res/map_small_edge.jpg",
                      fov=15,
                      food_spawn_threshold=255,
                      percent_for_game_over=100,
                      steps_for_game_over=100,
                      wait_key=1,
                      render=True)

    agent = DQNAgent(env)

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
        agent.tensorboard.step = episode

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

            # Render
            env.render_map()
            env.render_sub_map()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, learning_rate=agent.learning_rate)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


if __name__ == '__main__':
    main()
