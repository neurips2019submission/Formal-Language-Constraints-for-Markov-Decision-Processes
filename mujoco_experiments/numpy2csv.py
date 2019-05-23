import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'npys', help='List of npy files', nargs='*')
parser.add_argument('--keys', nargs="*", default=[])
args = parser.parse_args()
args.npys = [os.path.abspath(npy) for npy in args.npys]
print(args.npys)

data = []
with open('data.csv', 'w') as csv:
    for (i, npy) in enumerate(args.npys):
        strs = map(str, np.load(npy).tolist())
        key = args.keys[i] if len(args.keys) > i else ''
        csv.write(key + ',' + ','.join(strs) + '\n')
