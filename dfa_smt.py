from pysmt.shortcuts import Symbol, AllDifferent, Solver, And, get_model, NotEquals, Select, Equals, Not, Int, Or, Iff, Implies
from pysmt.typing import INT, BOOL, ArrayType
import numpy as np
import time

from networkx.algorithms import approximation as apx

from util import *

def graph_color_ass_smt(G, h):
    n = len(G.nodes)
    x = [Symbol(f"x_{i}", INT) for i in range(n)]
    f = And(
        *(0 < v for v in x),
        *(v <= h for v in x),
        *(NotEquals(x[u],x[v]) for u, v in G.edges)
    )
    model = get_model(f)
    if model:
        print(model)
    else:
        print("No solution found")

def min_dfa_setup_model(train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    m = len(EE)
    s = len(sigma)

    x = [Symbol(f"x_{i}", INT) for i in range(n)]
    y = [Symbol(f"y_{i}", ArrayType(INT, INT)) for i in range(s)]
    z = Symbol(f"z", ArrayType(INT, BOOL))

    constraints = []
    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                constraints.append(Equals(Select(y[l], x[pi]), x[ci]))

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            if accept == 1:
                constraints.append(Select(z, x[pi]))
            else:
                constraints.append(Not(Select(z, x[pi])))

    constraints.extend(0 < v for v in x)
    constraints.extend(v <= h for v in x)
    constraints.extend(NotEquals(x[u],x[v]) for u, v in G.edges)

    return x, y, z, constraints

def break_dfa_symmetry_bfs(x, y, sigma, prefixes_r, h):
    epsilon_i = prefixes_r[()]
    assert epsilon_i == 0
    constraints = []
    constraints.append(Equals(x[epsilon_i], Int(1)))

    t = [[Symbol(f"t_{i}_{j}", BOOL) for j in range(h)] for i in range(h)]
    p = [[Symbol(f"p_{i}_{j}", BOOL) for j in range(h)] for i in range(h)]

    for i, j in it.combinations(range(h), 2):
        constraints.append(Iff(
            t[i][j],
            Or(*(Equals(Select(y[l], Int(i+1)), Int(j+1)) for l in range(len(sigma))))
        ))

        constraints.append(Iff(
            p[j][i],
            And(
                t[i][j],
                *(Not(t[k][j]) for k in range(i))
            )
        ))

    for j in range(1, h):
        constraints.append(Or(*(p[j][k] for k in range(j))))

    for k, i, j in it.combinations(range(h-1), 3):
        constraints.append(Implies(p[j][i], Not(p[j+1][k])))

    assert len(sigma) == 2

    for i, j in it.combinations(range(h-1), 2):
        constraints.append(Implies(And(p[j][i], p[j+1][i]), Equals(Select(y[0], Int(i+1)), Int(j+1))))

    return constraints

def break_dfa_symmetry_clique(x, clique):
    constraints = []
    for i, c in enumerate(clique):
        constraints.append(Equals(x[c], Int(i+1)))
    return constraints


def extract_dfa(model, x, y, z, sigma, prefixes_r, h):
    epsilon_i = prefixes_r[()]
    q1 = np.zeros(h, np.int)
    q1[int(model.get_py_value(x[0])) - 1] = 1
    qinf = np.zeros(h, np.int)
    for i in range(h):
        qinf[i] = model.get_py_value(Select(z, Int(i) + 1))
    A = [np.zeros((h, h), np.int) for _ in sigma]
    for s in range(len(sigma)):
        for i in range(h):
            j = model.get_py_value(Select(y[s], Int(i) + 1)) - 1
            A[s][i][j] = 1

    return q1, qinf, A

def clique_decompose(G):
    cliques = []
    from copy import deepcopy
    G = deepcopy(G)
    while True:
        clique = apx.clique.max_clique(G)
        if len(clique) == 2:
            break
        cliques.append(clique)
        G.remove_edges_from(it.combinations(clique, 2))
    return cliques, G

if __name__ == '__main__':
    fname = 'dcts/dfa_9_try_3.dct'

    sigma = range(2)

    print(f"Learning {fname}")
    train = read_dct(fname)
    prefixes_f, prefixes_r, suffixes_f, suffixes_r = enumerate_fixes(train)
    G  = build_distinguishability_graph(train, prefixes_r, suffixes_r)

    clique = dfa_clique_approx(G, train, prefixes_r)
    states = len(clique)
    state_limit = 13
    total_time = 0

    while True:
        print(f'Building problem for {states} states')
        x, y, z, model_c = min_dfa_setup_model(train, prefixes_f, prefixes_r, G, sigma, states)
        symmetry_c = break_dfa_symmetry_bfs(x, y, sigma, prefixes_r, states)
        # symmetry_c = break_dfa_symmetry_clique(x, clique)
        start_time = time.perf_counter()
        print(f'Starting solver: {len(model_c) + len(symmetry_c)} constraints')
        model = get_model(And(*model_c, *symmetry_c).simplify(), 'z3')
        end_time = time.perf_counter()
        solve_time = end_time - start_time
        total_time += solve_time
        print(f"Took {solve_time:.4f} seconds")

        if model: break

        states += 1
        if states > state_limit: break

    print(f'Found solution with {states} states!')
    print(f'Took {total_time:.4f} seconds')

    dfa = extract_dfa(model, x, y, z, sigma, prefixes_r, states)

    # assert is_sorted(bfs_dfa(*dfa)[::1])
    assert all([dfa_eval(dfa, seq, sigma) == accept for seq, accept in train])
