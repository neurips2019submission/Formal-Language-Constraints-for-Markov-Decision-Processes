from dfa.regex2dfa.converter import Regex2DFA


class DFA:
    def __init__(self, reg_ex):
        self._regexer = Regex2DFA(reg_ex)
        self._current_state = 'q0'
        self._num_state = len(self._regexer._dfa.nodes())

        self.states = self._regexer._states
        self.accept_states = self._regexer._accept_states

    def try_step(self, action):
        is_accept, _ = self._regexer.traverse_dfa(action, self._current_state)
        return is_accept

    def simulate(self, action):
        is_accept, self._current_state = self._regexer.traverse_dfa(
            action, self._current_state)

        return is_accept

    def reset(self):
        self._current_state = 'q0'

    def num_states(self):
        return self._num_state
