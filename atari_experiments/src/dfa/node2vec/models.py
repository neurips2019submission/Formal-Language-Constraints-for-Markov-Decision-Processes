from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, InputLayer, Input, Concatenate, concatenate
import keras.backend as K
import numpy as np

def node2vec_model(input_shape, window_length, nb_actions, emb):
    """
    emb: pandas data frame
    """
    mdp_shape = (window_length,) + input_shape
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
    mdp_first_conv2d = Conv2D(32, (8, 8), activation='relu', strides=(4, 4))(mdp_permute)
    mdp_second_conv2d = Conv2D(64, (4, 4), activation='relu', strides=(2, 2))(mdp_first_conv2d)
    mdp_third_conv2d = Conv2D(64, (3, 3), activation='relu', strides=(1, 1))(mdp_second_conv2d)
    mdp_flatten = Flatten()(mdp_third_conv2d)
    mdp_first_dense = Dense(512, activation='relu')(mdp_flatten)

    second_input_shape = (emb.shape[1],)
    dfa_emb_input = Input(shape=second_input_shape, name='DFA_input_layer')
    merged = concatenate([mdp_first_dense, dfa_emb_input], axis=1)

    output = Dense(nb_actions, activation='linear')(merged)

    model = Model(inputs=[mdp_input, dfa_emb_input], outputs=output)

    return model
