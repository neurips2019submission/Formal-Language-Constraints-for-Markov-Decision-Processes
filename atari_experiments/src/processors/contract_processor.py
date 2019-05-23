import enum
import os

import numpy as np
from PIL import Image

import dfa.dfa as dfa
import rl.core
from processors.logging import StepMonitor, LogBuffer
from processors.util import identify_mothership

MODE = enum.Enum('Mode', 'punish halt off')
INPUT_SHAPE = (84, 84)


class ContractProcessor(rl.core.Processor):
    """Procssor which contaains a contract DFA and modifies reward when the DFA
    reaches a violating state.

    The DFA is stepped when `process_action` is called and reset if it's in a
    violating state when `process_reward` is called. The modes are 'off' for
    when you want to record violations, but not shape rewards, 'punish' for
    shaping rewards when a violation occurs, and 'halt' to shape rewards and
    stop the episode when a violation occurs.
    """

    def __init__(self, reg_ex, mode, log_root, nb_actions, enforce_contract, violation_reward, action_active=True, state_active=False, state_tl=None):
        if mode == 'punish':
            self.mode = MODE.punish
            self._violation_reward = violation_reward
        elif mode == 'halt':
            self.mode = MODE.halt
            self._violation_reward = violation_reward
        elif mode == 'off':
            self.mode = MODE.off
        self._dfa = dfa.DFA(reg_ex)
        self.state_active = state_active
        self.action_active = action_active
        self._enforce_contract = enforce_contract
        self._nb_actions = nb_actions
        self.step = 0
        self.step_monitor = StepMonitor(log_root)
        self.violation_monitor = LogBuffer(1024, (), dtype=bool)
        self.reward_mod_monitor = LogBuffer(1024, (), dtype=float)

        self._is_violated = False
        self._was_violated = False
        self._episode_number = 0
        self._episode_reward = 0
        self._episode_violations = 0
        self._total_violations = 0

        self._log_root = log_root
        ep_log_filename = log_root + '_episodes.csv'
        self._ep_log_file = open(ep_log_filename, 'w')

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tuple (observation, reward, done, reward) with with all elements after being processed.
        """
        self.step += 1
        observation = self.process_observation(observation)
        raw_reward = reward
        reward = self.process_reward(reward)
        self.reward_mod_monitor.log(reward - raw_reward)
        info = self.process_info(info)

        if self._is_violated:
            self._is_violated = False
            self._dfa.reset()
            self.violation_monitor.log(True)
            self._total_violations += 1
            self._episode_violations += 1
            if self.mode == MODE.halt:
                done = True
        else:
            self.violation_monitor.log(False)

        if done:
            self._dfa.reset()
            log = '{0},{1},{2}'.format(self._episode_number,
                                       self._episode_violations,
                                       self._episode_reward)
            self._ep_log_file.write(log + '\n')
            self._episode_reward = 0
            self._episode_violations = 0
            self._ep_log_file.flush()
            self._episode_number += 1
            self.step = 0

        # OBS HERE
        if identify_mothership(observation) != (0,0,0,0):
            pass

        return observation, reward, done, info

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert(
            'L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype(
            'uint8')  # saves storage in experience memory

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it. Resets the DFA if it was in a violating state.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """
        if self._is_violated and self.mode != MODE.off:
            reward = self._violation_reward
        self._episode_reward += reward
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            info (dict): An info as obtained by the environment
        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action, q_values):
        """Processes an action predicted by an agent but before execution in an environment.
        # Arguments
            action (int): Action given to the environment
        # Returns
            Processed action given to the environment
        """
        if self._enforce_contract:
            action_copy = action
            if q_values is None:
                q_values = np.random.rand(self._nb_actions)
                max_index = np.argmax(q_values)
                max_q_value = q_values[max_index]
                q_values[max_index] = q_values[action]
                q_values[action] = max_q_value

            while self._dfa.try_step(action):
                q_values[action] = float('-inf')
                action = np.argmax(q_values)

                if q_values[action] == float('-inf'):
                    action = action_copy
                    break

        self._is_violated = self._dfa.simulate(action)
        self._was_violated = self._is_violated

        return action

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.``
        return batch.astype('float32') / 255.

    def process_experience(self, observation, action, reward, done):
        self.step_monitor.update(observation, action, reward, done)
        return observation, action, reward, done

    def finalize(self):
        self.step_monitor.save()
        self.violation_monitor.save(os.path.join(self._log_root, 'violations'))
        self.reward_mod_monitor.save(os.path.join(self._log_root, 'reward_mod'))

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.
        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []
