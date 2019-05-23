import argparse
import json
import os

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

import models.atari_model as atari_model
import models.merged_model as merged_model
import processors.action_history_contract_processor as action_history_contract_processor
import processors.contract_processor as contract_processor
import processors.graph_emb_processor as graph_emb_processor
import processors.one_hot_counting as one_hot_counting
import processors.stateful_contract_processor as stateful_contract_processor
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

ENFORCE_CONTRACT = True
enforce_contract = ENFORCE_CONTRACT
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

VISUALIZE = False
VERBOSE = 2

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument(
    '--env-name',
    type=str,
)
parser.add_argument('--contract', type=str)
parser.add_argument('--architecture', type=str)
parser.add_argument('--contract-mode',
                    type=str,
                    choices=['off', 'punish', 'halt'])
parser.add_argument('--violation_reward', type=float, default=-1000)
parser.add_argument('--steps', type=int)
parser.add_argument('--train_seed', type=int)
parser.add_argument('--test_seed', type=int)
parser.add_argument('--emb', type=str)
parser.add_argument('--enforce_contract', type=bool, default=False)
parser.add_argument('--doom_scenario', type=str)
parser.add_argument('--log_root', type=str, default='./logs/')
parser.add_argument('--weights_root', type=str, default='./weights/')
args = parser.parse_args()


def output_args(log_dir, args):
    print(args)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.json'), 'w') as arg_file:
        args_copy = vars(args).copy()  # start with x's keys and values
        import subprocess
        args_copy['git_commit'] = subprocess.check_output(
            ["git", "describe", "--always"]).strip().decode("utf-8")
        json.dump(args_copy, arg_file)


def filename_prefix_fn(env_name,
                       contract,
                       architecture,
                       contract_mode,
                       steps,
                       train_seed,
                       test_seed=None):
    root = 'env={}-c={}-arc={}-mode={}-ns={}-seed={}'.format(
        env_name, contract, architecture, contract_mode, steps, train_seed)
    if test_seed is not None:
        root += '-test_seed=' + str(test_seed)
    return root


def build_dqn(env_name, contract, architecture, contract_mode, steps,
              nb_actions, emb, enforce_contract, log_prefix, violation_reward):
    # map from contract name to regex
    config = yaml.load(open('./src/pipeline/config.yaml'))
    contract = config[env_name][contract]['regular']

    if architecture == 'contract':
        # 1.) BASELINE WITH 0 AUGMENTATION (NO STATE NETWORK)
        # Model=default atari model; Processor=contract processor
        processor = contract_processor.ContractProcessor(
            reg_ex=contract,
            mode=contract_mode,
            log_root=log_prefix,
            nb_actions=nb_actions,
            enforce_contract=enforce_contract,
            violation_reward=violation_reward)
        model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions)

    elif architecture == 'contract_action_history':
        # 2.) BASELINE WITH CONSTANT ACTION HISTORY NETWORK
        # Model=merged model; Processor=contract processor
        ACTION_HISTORY_SIZE = 10
        processor = action_history_contract_processor.ContractProcessorWithActionHistory(
            reg_ex=contract,
            mode=contract_mode,
            log_root=log_prefix,
            nb_actions=nb_actions,
            enforce_contract=enforce_contract,
            violation_reward=violation_reward,
            action_history_size=ACTION_HISTORY_SIZE)
        dfa_input_shape = (
            ACTION_HISTORY_SIZE,
            nb_actions,
        )
        model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH,
                                          nb_actions, dfa_input_shape)

    elif architecture == 'contract_dfa_state':
        # 3.) DFA STATE NETWORK USING ONE-HOT
        # Model=merged model; Processor=contract processor
        processor = stateful_contract_processor.ContractProcessorWithState(
            reg_ex=contract,
            mode=contract_mode,
            log_root=log_prefix,
            nb_actions=nb_actions,
            enforce_contract=enforce_contract,
            violation_reward=violation_reward,
        )
        dfa_input_shape = (
            WINDOW_LENGTH,
            processor.get_num_states(),
        )
        model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH,
                                          nb_actions, dfa_input_shape)

    elif architecture == 'one_hot_counting':
        processor = one_hot_counting.OneHotCountingProcessor(
            reg_ex=contract,
            mode=contract_mode,
            log_root=log_prefix,
            nb_actions=nb_actions,
            enforce_contract=enforce_contract,
            violation_reward=violation_reward)
        dfa_input_shape = (
            WINDOW_LENGTH,
            processor.get_num_states(),
        )
        model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH,
                                          nb_actions, dfa_input_shape)

    elif architecture == 'contract_graph_emb':
        # 4.) DFA STATE USING NODE2VEC
        # Model=graph embedding model; Processor=contract processor
        if emb is not None:
            emb = pd.read_csv(emb,
                              header=-1,
                              index_col=0,
                              skiprows=1,
                              delimiter=' ')
        processor = graph_emb_processor.DFAGraphEmbeddingProcessor(
            reg_ex=contract,
            mode=contract_mode,
            log_root=log_prefix,
            nb_actions=nb_actions,
            enforce_contract=enforce_contract,
            violation_reward=violation_reward,
            emb=emb)
        model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH,
                                          nb_actions, (
                                              WINDOW_LENGTH,
                                              emb.shape[1],
                                          ))
    else:
        assert False, 'unknown architecture'

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=steps)

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    return dqn, processor


def build_env(env_name, scenario):
    if env_name == 'doom' and scenario is not None:
        from envs.doom import make_env as make_doom_env
        env = make_doom_env(scenario=scenario,
                            grayscale=False,
                            input_shape=INPUT_SHAPE)
    else:
        env = gym.make(env_name)
    return env


def train(env_name, contract, architecture, contract_mode, steps, train_seed,
          emb, enforce_contract, doom_scenario, violation_reward):
    env = build_env(env_name, doom_scenario)
    np.random.seed(train_seed)
    env.seed(train_seed)
    nb_actions = env.action_space.n

    filename_prefix = filename_prefix_fn(env_name, contract, architecture,
                                         contract_mode, steps, train_seed,
                                         None)
    log_prefix = os.path.join(args.log_root, filename_prefix)
    output_args(log_prefix, args)
    weights_prefix = os.path.join(args.weights_root, filename_prefix)

    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    if not os.path.exists(weights_prefix):
        os.makedirs(weights_prefix)

    dqn, processor = build_dqn(env_name, contract, architecture, contract_mode,
                               steps, nb_actions, emb, enforce_contract,
                               log_prefix, violation_reward)

    dqn.fit(env,
            nb_steps=steps,
            log_interval=10000,
            visualize=VISUALIZE,
            verbose=VERBOSE)
    weights_filename = weights_prefix + '_weights.h5f'
    processor.finalize()
    return dqn.save_weights(weights_filename, overwrite=True)


def test(env_name, contract, architecture, contract_mode, steps, train_seed,
         test_seed, emb, enforce_contract, doom_scenario, violation_reward):
    env = build_env(env_name, doom_scenario)
    np.random.seed(test_seed)
    env.seed(test_seed)
    nb_actions = env.action_space.n

    log_prefix = os.path.join(
        args.log_root,
        filename_prefix_fn(env_name, contract, architecture, contract_mode,
                           steps, train_seed, test_seed))
    weights_filename = os.path.join(
        args.weights_root,
        filename_prefix_fn(env_name, contract, architecture, contract_mode,
                           steps, train_seed, None)) + '_weights.h5f'

    dqn = build_dqn(env_name, contract, architecture, contract_mode, steps,
                    nb_actions, emb, enforce_contract, log_prefix,
                    violation_reward)
    dqn.load_weights(weights_filename)

    dqn.test(env, nb_episodes=100, visualize=False, nb_max_start_steps=100)


if __name__ == '__main__':
    if args.task == 'train':
        train(args.env_name, args.contract, args.architecture,
              args.contract_mode, args.steps, args.train_seed, args.emb,
              args.enforce_contract, args.doom_scenario, args.violation_reward)
    elif args.task == 'test':
        test(args.env_name, args.contract, args.architecture,
             args.contract_mode, args.steps, args.train_seed, args.test_seed,
             args.emb, args.enforce_contract, args.doom_scenario,
             args.violation_reward)
    else:
        assert False, 'unknown task'
