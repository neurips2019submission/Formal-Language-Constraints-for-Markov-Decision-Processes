from PIL import Image
import numpy as np
import enum

from rl.core import Processor

import contracts.dfa

from edward.processors import ContractProcessor

INPUT_SHAPE = (84, 84)
PUNISH_REWARD = -1000
MODE = enum.Enum('Mode', 'punish halt off')

class DFAGraphEmbeddingProcessor(ContractProcessor):
    def __init__(self, regex, emb):
        super(DFAGraphEmbeddingProcessor, self).__init__(regex)
        self.emb = emb

    def get_num_states(self):
        return self._dfa.num_states

    def process_experience(self, observation, action, reward, done):
        observation = self.process_observation(observation)
        return observation, action, reward, done

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        state_emb = np.array([self.emb.loc[self._dfa.currentState].values]) # converts shape to (1,n)
        return [processed_batch, state_emb]
