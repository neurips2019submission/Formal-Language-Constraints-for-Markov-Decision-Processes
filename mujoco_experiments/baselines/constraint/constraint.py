import itertools
from collections import Counter

import numpy as np

from baselines.constraint.dfa import DFA

id_fn = lambda x: x


class Constraint(DFA):
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 s_tl=id_fn,
                 a_tl=id_fn,
                 s_active=True,
                 a_active=True):
        super(Constraint, self).__init__(reg_ex)
        self.name = name
        self.violation_reward = violation_reward
        self.s_tl = s_tl
        self.a_tl = a_tl
        self.s_active = s_active
        self.a_active = a_active

    def step(self, obs, action, done):
        is_viol = False
        if self.s_active and self.a_active:
            is_viol = is_viol | super().step('s')
        if self.s_active:
            is_viol = is_viol | super().step(self.s_tl(obs))
        if self.a_active:
            is_viol = is_viol | super().step(self.a_tl(action))
        rew_mod = self.violation_reward if is_viol else 0.
        return is_viol, rew_mod

    def reset(self):
        return super().reset()


class CountingPotentialConstraint(Constraint):
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 gamma,
                 s_tl=id_fn,
                 a_tl=id_fn,
                 s_active=True,
                 a_active=True):
        super(CountingPotentialConstraint, self).__init__(
            name, reg_ex, violation_reward, s_tl, a_tl, s_active, a_active)
        self.episode_visit_count = Counter()
        self.visit_count = Counter(self.states())
        self.violation_count = Counter(self.accepting_states())
        self.gamma = gamma
        self.prev_state = self.current_state

    def get_state_potentials(self):
        potential = lambda s: self.violation_count[s] / self.visit_count[s]
        return {s: potential(s) for s in self.states()}

    def step(self, obs, action, done):
        is_viol, _ = super().step(obs, action, done)
        dfa_state = self.current_state
        self.episode_visit_count[dfa_state] += 1

        current_viol_propn = (
            self.violation_count[dfa_state] / self.visit_count[dfa_state])
        prev_viol_propn = (self.violation_count[self.prev_state] /
                           self.visit_count[self.prev_state])
        rew_mod = (self.gamma * current_viol_propn -
                   prev_viol_propn) * self.violation_reward
        if self.prev_state in self.accepting_states(): rew_mod = 0

        if is_viol:
            self.violation_count += self.episode_visit_count
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()
        if done:
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()

        self.prev_state = dfa_state
        return is_viol, rew_mod
