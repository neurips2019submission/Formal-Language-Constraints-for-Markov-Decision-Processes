from collections import Counter

import numpy as np
import PIL

from processors.stateful_contract_processor import ContractProcessorWithState

INPUT_SHAPE = (84, 84)


class OneHotCountingProcessor(ContractProcessorWithState):
    def __init__(self,
                 reg_ex,
                 mode,
                 log_root,
                 nb_actions,
                 enforce_contract,
                 violation_reward,
                 gamma=0.99):
        super(OneHotCountingProcessor,
              self).__init__(reg_ex, mode, log_root, nb_actions,
                             enforce_contract, violation_reward)
        self.episode_visit_count = Counter()
        self.visit_count = Counter(self._dfa.states)
        self.violation_count = Counter(self._dfa.accept_states)
        self.gamma = gamma
        self.prev_state = self._dfa._current_state
        self.counting_violation_tracker = False

    def get_all_dfa_state_potentials(self):
        potential = lambda s: self.violation_count[s] / self.visit_count[s]
        return {s: potential(s) for s in self._dfa.states}

    def step_count(self, done):
        dfa_state = self._dfa._current_state
        self.episode_visit_count[dfa_state] += 1

        if self._was_violated:
            self.violation_count += self.episode_visit_count
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()
        if done:
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()

        self.prev_state = dfa_state

    def process_step(self, observation, reward, done, info):
        outs = super(OneHotCountingProcessor, self).process_step(
            observation, reward, done, info)
        self.step_count(done)
        return outs

    def process_reward(self, reward):
        dfa_state = self._dfa._current_state

        current_viol_propn = (
            self.violation_count[dfa_state] / self.visit_count[dfa_state])
        prev_viol_propn = (self.violation_count[self.prev_state] /
                           self.visit_count[self.prev_state])
        rew_mod = (self.gamma * current_viol_propn -
                   prev_viol_propn) * self._violation_reward

        return reward + rew_mod
