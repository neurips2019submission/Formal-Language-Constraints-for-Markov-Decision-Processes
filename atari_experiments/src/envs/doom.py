'''
from: https://gist.github.com/404akhan/8ba3921c2133e4a63243fa19b5d63dbb
'''

import itertools
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import sys
import random
from collections import deque, namedtuple

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.squeeze(img)
        return img.astype('uint8')

def make_env(scenario, grayscale, input_shape):

    width, height = input_shape
    env_spec = gym.spec('ppaquette/' + scenario)
    env_spec.id = scenario #'DoomBasic-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=width, height=height, grayscale=grayscale)
    return e

#NOOP, SHOOT, RIGHT, LEFT = 0, 1, 2, 3
#VALID_ACTIONS = [0, 1, 2, 3]
