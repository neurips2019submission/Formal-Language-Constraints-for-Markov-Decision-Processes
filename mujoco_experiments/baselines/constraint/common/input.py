import tensorflow as tf


def constraint_state_placeholder(constraint, batch_size, name='ContrSt'):
    return tf.placeholder(shape=[batch_size], dtype=tf.int32, name=name)


def constraint_state_input(constraint, batch_size=None, name='ContrSt'):
    placeholder = constraint_state_placeholder(constraint, batch_size, name)
    return placeholder, tf.to_float(
        tf.one_hot(placeholder, constraint.num_states))
