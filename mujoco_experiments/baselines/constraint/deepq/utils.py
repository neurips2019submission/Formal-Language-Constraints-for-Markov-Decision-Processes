from baselines.common.tf_util import adjust_shape
from baselines.constraint.common.input import constraint_state_input
from baselines.deepq.utils import ObservationInput
import tensorflow as tf
import numpy as np


class ConstraintStateAugmentedInput(ObservationInput):
    def __init__(self, observation_space, constraints, name=None):
        super().__init__(observation_space)
        self.constraint_num_states = [c.num_states for c in constraints]
        self.constraint_state_phs = [
            constraint_state_input(c, name=c.name)[1] for c in constraints
        ]

    def get(self):
        return [
            super().get(),
        ] + self.constraint_state_phs

    def make_feed_dict(self, data):
        """
        Assumes data is an interable whose second entry is an iterable of
        integer constraint states.
        """
        assert data.shape[1] == 2, "Actual shape is: {}".format(data.shape)
        obs = np.array(data[0][0])
        constraints = np.array(data[0][1])

        feed_dict = super().make_feed_dict(obs)
        for i, ph in enumerate(self.constraint_state_phs):
            c_one_hot = np.zeros([1, self.constraint_num_states[i]])
            c_one_hot[0, constraints[i]] = 1
            feed_dict[ph] = c_one_hot
        return feed_dict

    def batch_size(self):
        return tf.shape(self.get()[0])[0]
