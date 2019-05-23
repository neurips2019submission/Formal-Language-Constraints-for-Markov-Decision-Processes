import collections
import requests
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as clustering
import matplotlib.pyplot as plt
import graphviz
import pygraphviz
import networkx as nx

import node2vec

from sys import argv
from networkx.drawing import nx_agraph
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D


def main(args):
    URL = 'http://localhost:8080'
    service = 'regex2dfa2dot'
    letter = 'q'
    regex = '[0-3]*(23){2}[0-3]*'

    regexer = Regexer(URL, service)
    dfa = regexer.to_dfa(regex)
    is_accept, next_state = regexer.traverse_dfa(dfa, '2')
    print(is_accept, next_state)
    is_accept, next_state = regexer.traverse_dfa(dfa, '3', start=next_state)
    print(is_accept, next_state)
    is_accept, next_state = regexer.traverse_dfa(dfa, '2', start=next_state)
    print(is_accept, next_state)
    is_accept, next_state = regexer.traverse_dfa(dfa, '3', start=next_state)
    print(is_accept, next_state)

def main2(args):
    URL = 'http://localhost:8080'
    service = 'regex2dfa2dot'
    letter = 'q' # to represent states along with state number e.g. q0 q1 etc.
    #regex = '[0-3]*(23){2}[0-3]*|[0-3]*(23)(0|1)[0-3]*'
    regex = '[0-3]*(23){2}[0-3]*'

    response = requests.get('{}/{}/{}/{}/'.format(URL, service, letter, regex))

    dot = response.json()['dfaInDot']

    # is_accept, next_state = traverse_dfa(dot, '2', 'q0')
    # print('Accept: {}, Next state: {}'.format(is_accept, next_state))
    # is_accept, next_state = traverse_dfa(dot, '3', 'q2')
    # print('Accept: {}, Next state: {}'.format(is_accept, next_state))

    dfa = graphviz.Source(dot, format='png')
    dfa.render(args[1])

    embs = args[2]
    p = 1.1
    q = 0.4
    num_walks = 200
    walk_length = 80
    emb_dim = 2
    window = 3
    iters = 1000
    node2vec.get_embeddings(dot, True, False, p, q, num_walks, walk_length, emb_dim, window, iters, embs, args[3])

    # plotting vectors
    data = np.genfromtxt(embs, skip_header=1, delimiter=' ', dtype=str)
    labels = data[:, 0]
    embs = data[:, 1:]
    embs = embs.astype(np.float32)
    xs = embs[:, 0]
    ys = embs[:, 1]
    if emb_dim > 2:
        zs = embs[:, 2]

    fig = plt.figure(figsize=(8,6))
    if emb_dim > 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys)

    for i in range(len(labels)):
        if emb_dim > 2:
            ax.text(xs[i], ys[i], zs[i], labels[i], size=8, color='r')
        else:
            ax.text(xs[i], ys[i], labels[i], size=8, color='r')

    ax.set_xlabel('Emb1')
    ax.set_ylabel('Emb2')
    if emb_dim > 2:
        ax.set_zlabel('Emb3')

    plt.savefig(args[4])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(121)

    plt.title('metric=cosine, linkage=single')
    dist_mat_cos = pdist(embs, metric='cosine')

    #np.savetxt(X=squareform(dist_mat_cos), fname='dist_cos.csv', fmt='%.4f', delimiter=',')
    df = pd.DataFrame(squareform(dist_mat_cos), index=labels, columns=labels)
    df.to_csv('dist_cos.csv', float_format='%.4f')
    Z = clustering.linkage(dist_mat_cos, method='single')
    dn = clustering.dendrogram(Z, labels=labels)
    ax.plot()

    ax2 = fig.add_subplot(122)
    plt.title('metric=euclidean, linkage=single')
    dist_mat_euc = pdist(embs, metric='euclidean')
    #np.savetxt(X=squareform(dist_mat_euc), fname='dist_euc.csv', fmt='%.4f', delimiter=',')
    df = pd.DataFrame(squareform(dist_mat_euc), index=labels, columns=labels)
    df.to_csv('dist_euc.csv', float_format='%.4f')
    Z = clustering.linkage(dist_mat_euc, method='single')
    dn = clustering.dendrogram(Z, labels=labels)
    ax2.plot()
    plt.savefig(args[5])

if __name__ == '__main__':
    main(argv)
