import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Permute, concatenate
from keras.models import Model


def merged_model(input_shape, window_length, nb_actions, second_input_shape):
    mdp_shape = (window_length, ) + input_shape
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
    mdp_first_conv2d = Conv2D(
        32, (8, 8), activation='relu', strides=(4, 4))(mdp_permute)
    mdp_second_conv2d = Conv2D(
        64, (4, 4), activation='relu', strides=(2, 2))(mdp_first_conv2d)
    mdp_third_conv2d = Conv2D(
        64, (3, 3), activation='relu', strides=(1, 1))(mdp_second_conv2d)
    mdp_flatten = Flatten()(mdp_third_conv2d)
    mdp_first_dense = Dense(512, activation='relu')(mdp_flatten)

    dfa_input = Input(shape=second_input_shape, name='DFA_input_layer')

    if np.array(second_input_shape).size == 1:
        merged = concatenate([mdp_first_dense, dfa_input], axis=1)
    elif np.array(second_input_shape).size > 1:
        dfa_flatten = Flatten()(dfa_input)
        merged = concatenate([mdp_first_dense, dfa_flatten], axis=1)
    else:
        assert False, 'Wrong second input dimension!'

    output = Dense(nb_actions, activation='linear')(merged)

    model = Model(inputs=[mdp_input, dfa_input], outputs=output)
    return model
