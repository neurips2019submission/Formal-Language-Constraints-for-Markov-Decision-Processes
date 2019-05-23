import numpy as np
import os

from gym.core import Wrapper


class LogBuffer(object):
    def __init__(self, buffer_size, buffer_shape, dtype=np.uint8):
        self.buffer = np.zeros((buffer_size, ) + buffer_shape, dtype)
        self.next_step = 0

    def log(self, item):
        try:
            self.buffer[self.next_step] = item
        except IndexError:
            self.buffer = np.concatenate(
                [self.buffer, np.zeros_like(self.buffer)])
            self.buffer[self.next_step] = item
        self.next_step += 1
        return self.next_step

    def save(self, name):
        np.save(name, self.buffer[:self.next_step - 1])

class StepMonitor(object):
    def __init__(self, directory, log_size=1024):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log_size = log_size
        self.action_log = None
        self.reward_log = LogBuffer(log_size, (), dtype=np.float32)
        self.done_log = LogBuffer(log_size, (), dtype=np.int32)

    def update(self, ob, act, rew, done):
        if self.action_log is None:
            self.action_log = LogBuffer(self.log_size, act.shape, dtype=np.int32)
        act_ns = self.action_log.log(act)
        rew_ns = self.reward_log.log(rew)
        don_ns = self.done_log.log(done)
        # assert that the logs are staying in step
        assert act_ns == rew_ns
        assert act_ns == don_ns

    def save(self):
        self.action_log.save(os.path.join(self.directory, 'action'))
        self.reward_log.save(os.path.join(self.directory, 'reward'))
        self.done_log.save(os.path.join(self.directory, 'done'))
