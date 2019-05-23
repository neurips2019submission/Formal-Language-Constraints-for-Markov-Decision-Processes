from __future__ import print_function

import json
import os
import string
import sys
import tempfile

import pygraphviz
from networkx.drawing import nx_agraph


class Regex2DFA(object):
    def __init__(self, regex, letter='q'):
        self.regex = regex
        self._f = tempfile.NamedTemporaryFile(mode='w+')
        command = 'java -jar src/java-lib/regex2dfa.jar "{}" {}'.format(
            regex, self._f.name)
        os.system(command)

        with open(self._f.name) as fname:
            dot = fname.read()
            print(dot, file=open('{}.dot'.format(self._f.name), 'w'))

        self._dfa = nx_agraph.from_agraph(pygraphviz.AGraph(dot))
        self._accept_states = [
            n for n in self._dfa.nodes()
            if self._dfa.nodes.data('shape')[n] == 'doublecircle'
        ]
        self._states = [n for n in self._dfa.nodes()]

    def traverse_dfa(self, char, start):
        """
        dfa_dot: dfa in graphviz dot file
        first return value shows if next state is an accept state
        second return value is the next state
        """
        # not sure the type of action from RL Agent
        # to make sure type casting is used

        # convert [1-2][0-9] | 3[0-5] to letter in the upper case alph.
        if int(char) >= 10 and int(char) <= 35:
            i = int(char) - 10
            char = '"{}"'.format(string.ascii_uppercase[i])

        dfa = self._dfa
        edges = dfa.edges.data('label')
        transitions = list(filter(lambda x: x[0] == start, edges))
        for transition in transitions:
            if transition[2] == str(char):
                next_state = transition[1]
                if next_state in self._accept_states:
                    return True, next_state
                else:
                    return False, next_state

        return False, 'q0'


def main(args):
    assert len(args) > 1
    regex2dfa = Regex2DFA(args[1])
    print(regex2dfa.traverse_dfa('0', 'q0'))
    print(regex2dfa.traverse_dfa('2', 'q0'))
    print(regex2dfa.traverse_dfa('3', 'q2'))


if __name__ == '__main__':
    main(sys.argv)
