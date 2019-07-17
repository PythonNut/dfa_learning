import networkx as nx
from pysat.solvers import Glucose4
from pysat.card import *
from collections import defaultdict
import itertools as it
import numpy as np
import operator as op
from functools import reduce

def read_dct(fname):
    results = []
    with open(fname) as f:
        f.readline()
        for line in f:
            accept, _, *seq = map(int, line.strip().split(' '))
            results.append((tuple(seq), accept))
    return results

def enumerate_fixes(train):
    prefixes = set()
    suffixes = set()
    for seq, accept in train:
        for cut in range(0, len(seq)+1):
            prefix, suffix = seq[:cut], seq[cut:]
            prefixes.add(prefix)
            suffixes.add(suffix)

    prefixes_f = list(prefixes)
    suffixes_f = list(suffixes)
    prefixes_f.sort()
    suffixes_f.sort()

    prefixes_r = {p: i for i, p in enumerate(prefixes_f)}
    suffixes_r = {p: i for i, p in enumerate(suffixes_f)}

    return prefixes_f, prefixes_r, suffixes_f, suffixes_r

def build_distinguishability_graph(train, prefixes_r, suffixes_r):
    G = nx.Graph()
    G.add_nodes_from(range(len(prefixes_r)))

    dmap = defaultdict(lambda: defaultdict(lambda: set()))
    for seq, accept in train:
        for cut in range(-1, len(seq)):
            prefix, suffix = seq[:cut], seq[cut:]
            pi, si = prefixes_r[prefix], suffixes_r[suffix]
            dmap[si][accept].add(pi)

    for suffix in dmap.values():
        if 0 in suffix and 1 in suffix:
            for p0, p1 in it.product(suffix[0], suffix[1]):
                G.add_edge(p0, p1)

    return G

class IndexConvertor(object):
    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset

    def __getitem__(self, i):
        if len(self.shape) == 1:
            assert 0 <= i < self.shape[0]
            return i + self.offset

        return int(np.ravel_multi_index(i, self.shape) + self.offset)

    def __call__(self, j):
        return np.unravel_index(j - self.offset, self.shape)

    def __str__(self):
        return f"I{self.shape}"

    __repr__ = __str__

    def size(self):
        return reduce(op.mul, self.shape)

class VariableDispenser(object):
    def __init__(self):
        self.offset = 1
        self.variables = []

    def dispense(self, shape):
        I = IndexConvertor(shape, self.offset)
        self.offset += I.size()
        self.variables.append(I)
        return I

    def __str__(self):
        return f"VD{self.variables}"

    __repr__ = __str__

    def size(self):
        return self.offset - 1

    def unflatten(self, flat):
        assert len(flat) == self.size()
        results = []
        i = 0
        for v in self.variables:
            A = np.zeros(v.shape, np.bool)
            for _ in range(v.size()):
                i += 1
                A[v(i)] = flat[i-1] > 0
            results.append(A)

        return results



def min_dfa_setup_model(M, V, train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    m = len(EE)
    s = len(sigma)

    x = V.dispense((n, h))
    y = V.dispense((s, h, h))
    z = V.dispense((h,))

    assert x[0, 0] == 1
    assert x[n-1, h-1] == y[0, 0, 0] - 1
    assert y[s-1, h-1, h-1] == z[0] - 1

    for v in VV:
        M.add_clause(x[v, i] for i in range(h))

    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                for i in range(h):
                    for j in range(h):
                        M.add_clause((y[l, i, j], -x[pi, i], -x[ci, j]))
                        M.add_clause((-y[l, i, j], -x[pi, i], x[ci, j]))

    for l, i, j, k in it.product(range(len(sigma)), *[range(h)]*3):
        if j < k:
            M.add_clause((-y[l, i, j], -y[l, i, k]))

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            if accept == 1:
                M.add_clause((-x[pi, i], z[i]))
            else:
                M.add_clause((-x[pi, i], -z[i]))

    return M

if __name__ == '__main__':
    train = read_dct('dcts/dfa_8_states_try_1.dct')
    prefixes_f, prefixes_r, suffixes_f, suffixes_r = enumerate_fixes(train)
    G  = build_distinguishability_graph(train, prefixes_r, suffixes_r)

    V = VariableDispenser()

    M = Glucose4()
    M = min_dfa_setup_model(M, V, train, prefixes_f, prefixes_r, G, range(2), 8)

    print(f"Starting solver: {M.nof_vars()} vars, {M.nof_clauses()} clauses")

    assert M.solve()

    x, y, z = V.unflatten(M.get_model())
