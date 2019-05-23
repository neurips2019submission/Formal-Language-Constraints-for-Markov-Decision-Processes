import collections
import requests
import pygraphviz

from networkx.drawing import nx_agraph

class Regexer(object):
    def __init__(self, URL, service, letter='q', initial='q0'):
        self._URL = URL
        self._service = service
        self._letter = letter
        self._initial = initial

    def to_dfa(self, regex):
        response = requests.get(
            '{}/{}/{}/{}/'.format(self._URL, self._service, self._letter, regex)
        )

        dot = response.json()['dfaInDot']
        dfa = nx_agraph.from_agraph(pygraphviz.AGraph(dot))

        accept_states = [n for n in dfa.nodes() if dfa.nodes.data('shape')[n] == 'doublecircle']

        dfa_triple = collections.namedtuple('DFA', ['dot', 'start', 'accepts'])

        return dfa_triple(dfa, self._initial, accept_states)

def traverse_dfa(dfa_triple, char, start):
    """
    dfa_triple: (dot: nx.multigraph, start: string, accepts: lists)
    first return value shows if next state is an accept state
    second return value is the next state
    """
    dfa = dfa_triple.dot # get the dfa in graph form
    edges = dfa.edges.data('label')
    transitions = list(filter(lambda x: x[0] == start, edges))
    for transition in transitions:
        if transition[2] == str(char):
            next_state = transition[1]
            if next_state in dfa_triple.accepts:
                return True, next_state
            else:
                return False, next_state

    return False, None
