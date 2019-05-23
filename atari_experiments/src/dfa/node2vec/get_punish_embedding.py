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
    with open(args[1], 'r') as f:
        dot = f.read()


    embs = args[2]
    p = 1
    q = 1
    num_walks = 200
    walk_length = 80
    emb_dim = 2
    window = 3
    iters = 1000
    node2vec.get_embeddings(dot, True, False, p, q, num_walks, walk_length, emb_dim, window, iters, embs, args[3])

if __name__ == '__main__':
    main(argv)
