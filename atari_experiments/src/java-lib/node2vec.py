from __future__ import print_function, division

import sys
import multiprocessing
import random

import numpy as np
import pygraphviz
import networkx as nx

from networkx.drawing import nx_agraph
from gensim.models import Word2Vec


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            # multiplexing node in neighbors list is needed since
            # there may be multiple edges between nodes with different labels
            # a -> b (edge label: 1)
            # a -> b (edge label: 2)
            # b is a's neighbor, but we need two bs in a's neighborhood for alias sampling
            num_of_edges_per_node = list(
                map(lambda x: len(G[cur][x].values()), G.neighbors(cur)))
            cur_nbrs = np.repeat(
                sorted(G.neighbors(cur)), num_of_edges_per_node)
            cur_nbrs = sorted(list(cur_nbrs))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0],
                                                    alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print('{}/{}'.format(walk_iter + 1, num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self.node2vec_walk(
                        walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            for key in G[dst][dst_nbr].keys():
                edge = G[dst][dst_nbr][key]
                if dst_nbr == src:
                    unnormalized_probs.append(edge['weight'] / p)
                elif G.has_edge(dst_nbr, src):
                    unnormalized_probs.append(edge['weight'])
                else:
                    unnormalized_probs.append(edge['weight'] / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs
        ]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            for nbr in sorted(G.neighbors(node)):
                unnormalized_probs = []
                for key in G[node][nbr].keys():
                    unnormalized_probs.append(G[node][nbr][key]['weight'])
                norm_const = sum(unnormalized_probs)
                normalized_probs = [
                    float(u_prob) / norm_const for u_prob in unnormalized_probs
                ]
                alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(
                    edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from non-uniform discrete distributions using alias sampling.
    '''
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def read_graph(dot_string, is_directed, is_weighted):
    '''
    Reads the input network in networkx.
    '''
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dot_string))
    for source, target in G.edges():
        for key in G[source][target].keys():
            G[source][target][key]['weight'] = 1

    if not is_directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, dimensions, window_size, iters, output):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=multiprocessing.cpu_count(),
        iter=iters)
    model.wv.save_word2vec_format(output)

    return


def get_embeddings(dot_string, is_directed, is_weighted, p, q, num_walks,
                   walk_length, dimensions, window_size, iters, output, log):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(dot_string, is_directed, is_weighted)
    G = Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    sorted_walks = sorted(walks, key=lambda x: x[0])
    i = 1
    initial = sorted_walks[0][0]
    for w in sorted_walks:
        if w[0] != initial:
            i = 1
            initial = w[0]
            print('{} {}'.format(i, ' '.join(w)), file=open(log, 'a'))
        else:
            print('{} {}'.format(i, ' '.join(w)), file=open(log, 'a'))
        i += 1
    learn_embeddings(walks, dimensions, window_size, iters, output)


def main(args):
    dot_string = open(args[1], 'r').read()
    output = args[2]
    p = 1
    q = 1
    dimension = 3
    window = 5
    get_embeddings(dot_string, True, False, p, q, 200, 80, dimension, window,
                   200, output, 'run.log')


if __name__ == '__main__':
    main(sys.argv)
