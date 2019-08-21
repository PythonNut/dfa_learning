import operator as op
from functools import reduce
from collections import defaultdict
import itertools as it
from functools import reduce, partial
from queue import Queue

import networkx as nx
import numpy as np
from pysat.solvers import *
from pysat.card import *

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

    def top_var(self):
        return self.size()

    def unflatten(self, flat):
        assert len(flat) <= self.size()
        results = []
        i = 0
        for v in self.variables:
            A = np.zeros(v.shape, np.bool)
            for _ in range(v.size()):
                i += 1
                j = v(i)
                if i > len(flat):
                    A[j] = False
                else:
                    A[v(i)] = flat[i-1] > 0
            results.append(A)

        return results

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

def add_equals_1(M, V, lits):
    if isinstance(M, Minicard):
        M.add_atmost(lits, 1)
        M.add_clause(lits)
        return
    top_id = V.top_var()
    encoding = EncType.seqcounter
    if len(lits) <= 6000:
        encoding = EncType.pairwise

    cnf = CardEnc.equals(
        lits=lits, encoding=encoding,
        top_id=top_id
    )
    new_vars = cnf.nv - top_id
    if new_vars > 0:
        V.dispense((new_vars,))
    M.append_formula(cnf)

def dfa_eval(dfa, seq, sigma):
    q1, qinf, A = dfa
    sigma_r = {s: i for i, s in enumerate(sigma)}
    return reduce(op.matmul, it.chain(
        (q1.transpose(),),
        (A[sigma_r[s]] for s in seq),
        (qinf,)
    ))

def transition_matrix_to_delta(A):
    k = A[0].shape[0]
    delta = {}
    for i in range(k):
        delta[i] = [np.where(a[i, :] == 1)[0][0] for a in A]
    return delta

def is_sorted(l):
    return all(a <= b for a, b in zip(l, l[1:]))

def bfs_dfa(q1, qinf, A):
    k = len(q1)
    delta = transition_matrix_to_delta(A)
    visited = np.zeros(k, np.bool)
    q = Queue()
    q1i = np.where(q1)[0][0]
    visited[q1i] = True
    result = []

    q.put(q1i)
    result.append(q1i)
    while not q.empty():
        v = q.get()
        for w in delta[v]:
            if not visited[w]:
                visited[w] = True
                q.put(w)
                result.append(w)

    return result

def print_noam(q1, qinf, A, fname='noam.txt'):
    with open(fname, 'w') as f:
        q1i = np.where(q1)[0][0]
        delta = transition_matrix_to_delta(A)
        f.writelines(['#states\n'])
        f.writelines([f's{i}\n' for i in range(len(q1))])
        f.writelines([f'#initial\ns{q1i}\n'])
        f.writelines(['#accepting\n'])
        f.writelines([f's{i}\n' for i in np.where(qinf)[0]])
        f.writelines(['#alphabet\n'])
        f.writelines([f'{i}\n' for i in range(len(A))])
        f.writelines(['#transitions\n'])
        f.writelines([f's{i}:{j}>s{o}\n' for i, l in delta.items() for j,o in enumerate(l)])

def greedy_clique(G):
    deg = G.degree()
    clique = set([max(deg, key=op.itemgetter(1))[0]])
    while True:
        common = reduce(op.and_, (set(G[v]) for v in clique))
        if not common:
            break

        clique.add(
            max(common, key=partial(op.getitem, deg))
        )

    return clique

def dfa_clique_approx(G, train, prefixes_r):
    pos = [prefixes_r[seq] for seq, accept in train if accept == 1]
    neg = [prefixes_r[seq] for seq, accept in train if accept == 0]
    posG, negG = G.subgraph(pos), G.subgraph(neg)
    pnC =  greedy_clique(posG) | greedy_clique(negG)
    allC = greedy_clique(G)

    return max(pnC, allC, key=len)
