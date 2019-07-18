import networkx as nx
from pysat.solvers import *
from pysat.card import *
from collections import defaultdict
import itertools as it
import numpy as np
import operator as op
from functools import reduce, partial
from queue import Queue

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

def add_equals_1(M, V, lits):
    top_id = V.top_var()
    encoding = EncType.seqcounter
    if len(lits) <= 6:
        encoding = EncType.pairwise

    cnf = CardEnc.equals(
        lits=lits, encoding=encoding,
        top_id=top_id
    )
    new_vars = cnf.nv - top_id
    if new_vars > 0:
        V.dispense((new_vars,))
    M.append_formula(cnf)

def min_dfa_setup_model(M, V, train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    m = len(EE)
    s = len(sigma)

    x = V.dispense((n, h))
    y = V.dispense((s, h, h))
    z = V.dispense((h,))

    for v in VV:
        add_equals_1(M, V, [x[v, i] for i in range(h)])

    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                for i in range(h):
                    for j in range(h):
                        M.add_clause((y[l, i, j], -x[pi, i], -x[ci, j]))
                        M.add_clause((-y[l, i, j], -x[pi, i], x[ci, j]))

    for l, i in it.product(range(len(sigma)), range(h)):
        add_equals_1(M, V, [y[l, i, j] for j in range(h)])

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            if accept == 1:
                M.add_clause((-x[pi, i], z[i]))
            else:
                M.add_clause((-x[pi, i], -z[i]))

    for u, v in EE:
        for i in range(h):
            M.add_clause((-x[u, i], -x[v, i]))

    return M

def min_dfa_setup_model_pop(M, V, train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    s = len(sigma)

    a = V.dispense((h, n))
    b = V.dispense((n, h))
    y = V.dispense((s, h, h))
    z = V.dispense((h,))

    q = 0

    for v in VV:
        M.add_clause((-b[v, 0],))
        M.add_clause((-a[h-1, v],))

        for i in range(h-1):
            M.add_clause((-a[i+1, v], a[i, v]))

            M.add_clause((a[i, v], b[v, i+1]))
            M.add_clause((-a[i, v], -b[v, i+1]))

            M.add_clause((-a[i, v], a[i, q]))

    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                for i in range(h):
                    for j in range(h):
                        M.add_clause((y[l, i, j], a[i, pi], b[pi, i], a[j, ci], b[ci, j]))
                        M.add_clause((-y[l, i, j], a[i, pi], b[pi, i], -a[j, ci]))
                        M.add_clause((-y[l, i, j], a[i, pi], b[pi, i], -b[ci, j]))

    for l, i in it.product(range(len(sigma)), range(h)):
        add_equals_1(M, V, [y[l, i, j] for j in range(h)])

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            if accept == 1:
                M.add_clause((a[i, pi], b[pi, i], z[i]))
            else:
                M.add_clause((a[i, pi], b[pi, i], -z[i]))

    for u, v in EE:
        for i in range(h):
            M.add_clause((a[i, u], b[u, i], a[i, v], b[v, i]))

    return M

def min_dfa_setup_model_pop2(M, V, train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    s = len(sigma)

    a = V.dispense((h, n))
    b = V.dispense((n, h))
    x = V.dispense((n, h))
    y = V.dispense((s, h, h))
    z = V.dispense((h,))

    q = 0

    for v in VV:
        add_equals_1(M, V, [x[v, i] for i in range(h)])

        M.add_clause((-b[v, 0],))
        M.add_clause((-a[h-1, v],))

        for i in range(h):
            add_equals_1(M, V, (x[v, i], a[i, v], b[v, i]))

        for i in range(h-1):
            M.add_clause((-a[i+1, v], a[i, v]))

            M.add_clause((a[i, v], b[v, i+1]))
            M.add_clause((-a[i, v], -b[v, i+1]))

            M.add_clause((-a[i, v], a[i, q]))

    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                for i in range(h):
                    for j in range(h):
                        M.add_clause((y[l, i, j], -x[pi, i], -x[ci, j]))
                        M.add_clause((-y[l, i, j], -x[pi, i], x[ci, j]))

    for l, i in it.product(range(len(sigma)), range(h)):
        add_equals_1(M, V, [y[l, i, j] for j in range(h)])

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            if accept == 1:
                M.add_clause((-x[pi, i], z[i]))
            else:
                M.add_clause((-x[pi, i], -z[i]))

    for u, v in EE:
        for i in range(h):
            M.add_clause((-x[u, i], -x[v, i]))

    return M

def break_dfa_symmetry_bfs(M, V, sigma, prefixes_r, h):
    x, y, z, *_ = V.variables
    s = len(sigma)

    p = V.dispense((h, h))
    t = V.dispense((h, h))

    epsilon_i = prefixes_r[()]
    assert epsilon_i == 0
    M.add_clause((x[epsilon_i, 0],))

    for i, j in it.combinations(range(h), 2):
        M.add_clause(it.chain((-t[i, j],), (y[l, i, j] for l in range(s))))
        for l in range(s):
            M.add_clause((-y[l, i, j], t[i, j]))

        M.add_clause((-p[j, i], t[i, j]))
        M.add_clause(it.chain(
            (-t[i, j], p[j, i]),
            (t[k, j] for k in range(i))
        ))

    for k, i, j in it.combinations(range(h), 3):
        M.add_clause((-p[j, i], -t[k, j]))

    for j in range(1, h):
        M.add_clause(p[j, i] for i in range(j))

    for k, i, j in it.combinations(range(h-1), 3):
        M.add_clause((-p[j, i], -p[j+1, k]))

    if s == 2:
        for i, j in it.combinations(range(h-1), 2):
            M.add_clause((-p[j, i], -p[j+1, i], y[0, i, j]))
    else:
        m = V.dispense((s, h, h))

        for i, j in it.combinations(range(h), 2):
            for l in range(s):
                M.add_clause((-m[l, i, j], y[l, i ,j]))
                M.add_clause(it.chain(
                    (-y[l, i, j], m[l, i, j]),
                    (y[k, i, j] for k in range(l))
                ))

            for k, n in it.combinations(range(s), 2):
                M.add_clause((-m[n, i, j], -y[k, i, j]))

        for i, j in it.combinations(range(h-1), 2):
            for k, n in it.combinations(range(s), 2):
                M.add_clause((-p[j, i], -p[j+1, i], -m[n, i, j], -m[k, i, j+1]))

    return M

def break_dfa_symmetry_bfs_pop(M, V, sigma, prefixes_r, h):
    a, b, y, z, *_ = V.variables
    s = len(sigma)

    p = V.dispense((h, h))
    t = V.dispense((h, h))

    epsilon_i = prefixes_r[()]
    assert epsilon_i == 0
    M.add_clause((-a[h-1, epsilon_i],))
    M.add_clause((-b[epsilon_i, h-1],))

    for j, i in it.combinations(range(h), 2):
        M.add_clause(it.chain((-t[i, j],), (y[l, i, j] for l in range(s))))
        for l in range(s):
            M.add_clause((-y[l, i, j], t[i, j]))

        M.add_clause((-p[j, i], t[i, j]))
        M.add_clause(it.chain(
            (-t[i, j], p[j, i]),
            (t[k, j] for k in range(i+1, h))
        ))

    for j, i, k in it.combinations(range(h), 3):
        M.add_clause((-p[j, i], -t[k, j]))

    for j in range(h-1):
        M.add_clause(p[j, i] for i in range(j+1, h))

    for j, i, k in it.combinations(range(1, h), 3):
        M.add_clause((-p[j, i], -p[j-1, k]))

    if s == 2:
        for j, i in it.combinations(range(1, h), 2):
            M.add_clause((-p[j, i], -p[j-1, i], y[0, i, j]))
    else:
        assert False
        m = V.dispense((s, h, h))

        for i, j in it.combinations(range(h), 2):
            for l in range(s):
                M.add_clause((-m[l, i, j], y[l, i ,j]))
                M.add_clause(it.chain(
                    (-y[l, i, j], m[l, i, j]),
                    (y[k, i, j] for k in range(l))
                ))

            for k, n in it.combinations(range(s), 2):
                M.add_clause((-m[n, i, j], -y[k, i, j]))

        for i, j in it.combinations(range(h-1), 2):
            for k, n in it.combinations(range(s), 2):
                M.add_clause((-p[j, i], -p[j+1, i], -m[n, i, j], -m[k, i, j+1]))

    return M

def break_dfa_symmetry_bfs_pop2(M, V, sigma, prefixes_r, h):
    _, _, x, y, z, *_ = V.variables
    s = len(sigma)

    p = V.dispense((h, h))
    t = V.dispense((h, h))

    epsilon_i = prefixes_r[()]
    assert epsilon_i == 0
    M.add_clause((x[epsilon_i, h-1],))

    for j, i in it.combinations(range(h), 2):
        M.add_clause(it.chain((-t[i, j],), (y[l, i, j] for l in range(s))))
        for l in range(s):
            M.add_clause((-y[l, i, j], t[i, j]))

        M.add_clause((-p[j, i], t[i, j]))
        M.add_clause(it.chain(
            (-t[i, j], p[j, i]),
            (t[k, j] for k in range(i+1, h))
        ))

    for j, i, k in it.combinations(range(h), 3):
        M.add_clause((-p[j, i], -t[k, j]))

    for j in range(h-1):
        M.add_clause(p[j, i] for i in range(j+1, h))

    for j, i, k in it.combinations(range(1, h), 3):
        M.add_clause((-p[j, i], -p[j-1, k]))

    if s == 2:
        for j, i in it.combinations(range(1, h), 2):
            M.add_clause((-p[j, i], -p[j-1, i], y[0, i, j]))
    else:
        assert False
        m = V.dispense((s, h, h))

        for i, j in it.combinations(range(h), 2):
            for l in range(s):
                M.add_clause((-m[l, i, j], y[l, i ,j]))
                M.add_clause(it.chain(
                    (-y[l, i, j], m[l, i, j]),
                    (y[k, i, j] for k in range(l))
                ))

            for k, n in it.combinations(range(s), 2):
                M.add_clause((-m[n, i, j], -y[k, i, j]))

        for i, j in it.combinations(range(h-1), 2):
            for k, n in it.combinations(range(s), 2):
                M.add_clause((-p[j, i], -p[j+1, i], -m[n, i, j], -m[k, i, j+1]))

    return M

def graph_color_ass(G, h):
    M = Glucose4(use_timer=True)
    V = VariableDispenser()

    n = len(G.nodes)
    x = V.dispense((n, h))

    for v in G.nodes:
        add_equals_1(M, V, [x[v, i] for i in range(h)])

    for u, v in G.edges:
        for i in range(h):
            M.add_clause((-x[u, i], -x[v, i]))

    solve = M.solve()
    print(f"Took {M.time():.4f} seconds")
    if not solve:
        return None

    x, *_ = V.unflatten(M.get_model())

    return x.nonzero()[1] + 1

def graph_color_pop(G, h):
    M = Glucose4(use_timer=True)
    V = VariableDispenser()

    n = len(G.nodes)
    y = V.dispense((h, n))
    z = V.dispense((n, h))

    q = 0

    for v in G.nodes:
        M.add_clause((-z[v, 0],))
        M.add_clause((-y[h-1, v],))

        for i in range(h-1):
            M.add_clause((-y[i+1, v], y[i, v]))

            M.add_clause((y[i, v], z[v, i+1]))
            M.add_clause((-y[i, v], -z[v, i+1]))

            M.add_clause((-y[i, v], y[i, q]))

    for u, v in G.edges:
        for i in range(h):
            M.add_clause((y[i, u], z[u, i], y[i, v], z[v, i]))

    solve = M.solve()
    print(f"Took {M.time():.4f} seconds")
    if not solve:
        return None

    y, z, *_ = V.unflatten(M.get_model())

    return np.sum(y, axis=0)

def graph_color_pop2(G, h):
    M = Glucose4(use_timer=True)
    V = VariableDispenser()

    n = len(G.nodes)
    y = V.dispense((h, n))
    z = V.dispense((n, h))
    x = V.dispense((n, h))

    q = 0

    for v in G.nodes:
        M.add_clause((-z[v, 0],))
        M.add_clause((-y[h-1, v],))

        for i in range(h):
            add_equals_1(M, V, (x[v, i], y[i, v], z[v, i]))

        for i in range(h-1):
            M.add_clause((-y[i+1, v], y[i, v]))

            M.add_clause((y[i, v], z[v, i+1]))
            M.add_clause((-y[i, v], -z[v, i+1]))

            M.add_clause((-y[i, v], y[i, q]))

    for u, v in G.edges:
        for i in range(h):
            M.add_clause((-x[u, i], -x[v, i]))

    solve = M.solve()
    print(f"Took {M.time():.4f} seconds")
    if not solve:
        return None

    y, z, x, *_ = V.unflatten(M.get_model())

    return x.nonzero()[1] + 1

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

def extract_dfa(M, V, sigma, prefixes_r):
    x, y, z, *_ = V.unflatten(M.get_model())
    epsilon_i = prefixes_r[()]
    A = [y[l, :, :].astype(int) for l in range(len(sigma))]
    q1 = x[epsilon_i, :].astype(int)
    return q1, z.astype(int), A

def extract_dfa_pop(M, V, sigma, prefixes_r):
    a, b, y, z, *_ = V.unflatten(M.get_model())
    x = 1 - (a.transpose() + b)
    epsilon_i = prefixes_r[()]
    A = [y[l, :, :].astype(int) for l in range(len(sigma))]
    q1 = x[epsilon_i, :].astype(int)
    return q1, z.astype(int), A

def extract_dfa_pop2(M, V, sigma, prefixes_r):
    _, _, x, y, z, *_ = V.unflatten(M.get_model())
    epsilon_i = prefixes_r[()]
    A = [y[l, :, :].astype(int) for l in range(len(sigma))]
    q1 = x[epsilon_i, :].astype(int)
    return q1, z.astype(int), A

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

if __name__ == '__main__':
    train = read_dct('dcts/dfa_12_try_6.dct')
    prefixes_f, prefixes_r, suffixes_f, suffixes_r = enumerate_fixes(train)
    G  = build_distinguishability_graph(train, prefixes_r, suffixes_r)

    sigma = range(2)

    states = len(dfa_clique_approx(G, train, prefixes_r))
    state_limit = 30

    total_time = 0

    while True:
        V = VariableDispenser()
        M = Glucose4(use_timer=True) # Glucose4(use_timer=True)

        print(f'Building problem for {states} states')
        min_dfa_setup_model_pop2(M, V, train, prefixes_f, prefixes_r, G, sigma, states)
        break_dfa_symmetry_bfs_pop2(M, V, sigma, prefixes_r, states)

        print(f'Starting solver: {M.nof_vars()} vars, {M.nof_clauses()} clauses')

        solve = M.solve()

        time = M.time()
        total_time += time
        print(f'Took {time:.4f} seconds')

        if solve:
            break
        else:
            M.delete()

        states += 1
        if states > state_limit: break

    print(f'Found solution with {states} states!')
    print(f'Took {total_time:.4f} seconds')

    dfa = extract_dfa_pop2(M, V, sigma, prefixes_r)

    assert is_sorted(bfs_dfa(*dfa)[::-1])
    assert all([dfa_eval(dfa, seq, range(2)) == accept for seq, accept in train])
