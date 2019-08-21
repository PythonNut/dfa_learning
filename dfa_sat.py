import networkx as nx
from pysat.solvers import *
from pysat.card import *
import itertools as it
import numpy as np
import operator as op

from util import *

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

def constrain_early_states_bfs(M, V, prefixes_f, h, sigma):
    s = len(sigma)
    x, *_ = V.variables

    for seq in prefixes_f:
        l = len(seq) + 1
        if s**l - 1 < h and l > 1:
            pi = prefixes_r[seq]
            print(seq, s**l - 1)
            for i in range(s**l - 1, h):
                M.add_clause((-x[pi, i],))

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

if __name__ == '__main__':
    fname = 'dcts/dfa_8_try_1.dct'

    print(f"Learning {fname}")
    train = read_dct(fname)
    prefixes_f, prefixes_r, suffixes_f, suffixes_r = enumerate_fixes(train)
    G  = build_distinguishability_graph(train, prefixes_r, suffixes_r)

    sigma = range(2)

    states = len(dfa_clique_approx(G, train, prefixes_r))
    state_limit = 13

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
