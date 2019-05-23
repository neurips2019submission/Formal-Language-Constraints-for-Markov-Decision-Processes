from __future__ import print_function
from __future__ import division

try:
    import IPython.core.ultratb as ultratb
    import sys
    sys.excepthook = ultratb.ColorTB()
except:
    pass

import collections
import enum
import argparse
import logging

import requests
import numpy as np
import pandas as pd
import gym
import keras.backend as K

from PIL import Image
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Permute, concatenate, multiply
from keras.optimizers import Adam
from keras.utils import print_summary
from keras.utils.vis_utils import model_to_dot

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from regexer import Regexer, traverse_dfa

# Logger
LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}
level = LEVELS.get('debug', logging.NOTSET)

sep = ' '*3
logging.basicConfig(
    level=level,
    format='%(asctime)-25s %(levelname)-10s %(message)s'.format(sep=sep),
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.FileHandler('run.log'),
        logging.StreamHandler()
    ]
)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

Mode = enum.Enum('Mode', 'punish halt off')
PUNISH_REWARD = -1000

class GraphEmbeddingProcessor(Processor):
    def __init__(self, dfa, emb, current_state='q0', mode=Mode.halt):
        self._dfa = dfa
        self._emb = emb # pandas data frame
        self._current_state = current_state
        self._is_violated = False
        self._mode = mode
        self._actions = []
        self._dfa_states = []
        self._episode = 1

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE

        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        state_emb = self._emb.loc[self._current_state].values
        state_emb = state_emb.reshape((1, ) + state_emb.shape)

        return [processed_batch, state_emb]

    def process_step(self, observation, reward, done, info):
        """
        Choosing to reset the DFA here works on the assumption that this is
        called after every `(a,r,s)` tuple has been returned from the MDP.
        """
        # process MDP outputs
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        # catch and record violations
        if self._is_violated:
            logging.info('Contract violation')
            logging.debug('Episode: {}'.format(self._episode))
            logging.debug('DFA state: {}'.format(self._current_state))

            if self._mode == Mode.halt:
                logging.info('Ending episode!!!')
                logging.debug('Actions: {}'.format(self._actions))
                done = True

        if done:
            self._current_state = 'q0' # reset
            self._episode += 1
            self._actions = []

        return observation, reward, done, info

    def process_reward(self, reward):
        if self._current_state in self._dfa.accepts and not self._mode == self._mode.off:
            reward = PUNISH_REWARD
            logging.info('Punishing for contract violation')

        return reward

    def process_action(self, action):
        """
        Choosing to advance the DFA here works on the assumption that this is
        called once every time the action will also be put into the MDP.
        """
        is_accept, self._current_state = traverse_dfa(
            self._dfa, action, self._current_state
        )
        self._is_violated = is_accept
        self._actions.append(action)
        self._dfa_states.append(self._current_state)
        return action

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--dfa-emb', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
# Horrible coding practice to define the model
# Layers in Sequential model was not visible in the model image
# I needed to define each layer explicitly to see in the model image

mdp_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE
# Dfa Embedding
if args.dfa_emb is not None:
    df_emb = pd.read_csv(args.dfa_emb, header=-1, index_col = 0, skiprows=1, delimiter=' ')
    dfa_emb_shape = (df_emb.shape[1],)


mdp_input = Input(shape=mdp_shape, name='MDP')
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    mdp_permute = Permute((2, 3, 1), input_shape=mdp_shape)
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    mdp_permute = Permute((1, 2, 3), input_shape=mdp_shape)
else:
    raise RuntimeError('Unknown image_dim_ordering.')

mdp_permute = mdp_permute(mdp_input)
mdp_first_conv2d = Conv2D(32, (8,8), activation='relu', strides=(4,4))(mdp_permute)
mdp_second_conv2d = Conv2D(64, (4,4), activation='relu', strides=(2,2))(mdp_first_conv2d)
mdp_third_conv2d = Conv2D(64, (3,3), activation='relu', strides=(1,1))(mdp_second_conv2d)
mdp_flatten = Flatten()(mdp_third_conv2d)
mdp_first_dense = Dense(512, activation='relu')(mdp_flatten)

dfa_input = Input(shape=dfa_emb_shape, name='DFA_EMB')
#selector_input = Input(shape=dfa_emb_shape, name='StateSelector')
#elem_mult = multiply([dfa_input, selector_input], trainable=False, name='Elem-wise')
#dfa_flatten = Flatten()(dfa_input)

merged = concatenate([mdp_first_dense, dfa_input], axis=1)
output = Dense(nb_actions, activation='linear')(merged)

model = Model(inputs=[mdp_input, dfa_input], outputs=output)
print_summary(model)
model_dot = model_to_dot(model, show_shapes=True, show_layer_names=True)
print(model_dot, file=open('model_arch.dot', 'w+'))

# Regexer
URL = 'http://localhost:8080'
service = 'regex2dfa2dot'
letter = 'q'
regex = '[0-3]*(23){2}[0-3]*'
regexer = Regexer(URL, service)

dfa_triple = regexer.to_dfa(regex)

# Finally, we configure and compile our agent. You can use every
# built-in Keras optimizer and even the metrics!

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

processor = GraphEmbeddingProcessor(dfa_triple, df_emb)

# Select a policy. We use eps-greedy action selection, which means
# that a random action is selected with probability eps. We anneal eps
# from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then
# gradually sticks to what it knows (low eps). We also set a dedicated
# eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures
# that the agent cannot get stuck.

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                              value_max=1., value_min=.1,
                              value_test=.05, nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and
# an on-going research topic.  If you want, you can experiment with
# the parameters or use a different policy. Another popular one is
# Boltzmann-style exploration: policy = BoltzmannQPolicy(tau=1.)  Feel
# free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
               memory=memory, processor=processor,
               nb_steps_warmup=50000, gamma=.99, # ###########################################3
               target_model_update=10000, train_interval=4,
               delta_clip=1.0, batch_size=1)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt
    # exception so that training can be prematurely aborted. Notice
    # that you can the built-in Keras callbacks!

    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=False)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
