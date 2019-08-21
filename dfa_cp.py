from ortools.sat.python import cp_model

from util import *
from collections import defaultdict

def flip_map(D):
    F = defaultdict(set)
    for k, v in D.items():
        F[v].add(k)
    return F

def graph_color(G, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    m = len(EE)

    model = cp_model.CpModel()

    x = [model.NewIntVar(0, h-1, f"x_{i}") for i in range(n)]

    for u, v in G.edges:
        model.Add(x[u] != x[v])

    # for k, v in flip_map(nx.greedy_color(nx.complement(G))).items():
    #     if len(v) > 2:
    #         model.AddAllDifferent([x[i] for i in v])

    return model, x

def min_dfa_setup_model(model, train, prefixes_f, prefixes_r, G, sigma, h):
    VV = G.nodes
    EE = G.edges

    n = len(VV)
    m = len(EE)
    s = len(sigma)

    model = cp_model.CpModel()

    x = [model.NewIntVar(0, h-1, f"x_{i}") for i in range(n)]
    y = [[model.NewIntVar(0, h-1, f"y_{i}_{j}") for j in range(h)] for i in range(s)]
    z = [model.NewBoolVar(f"z") for _ in range(h)]

    for pi, p in enumerate(prefixes_f):
        for l in range(len(sigma)):
            c = tuple([*p, sigma[l]])
            if c in prefixes_r:
                ci = prefixes_r[c]
                model.AddElement(x[pi], y[l], x[ci])

    for (seq, accept) in train:
        pi = prefixes_r[seq]
        for i in range(h):
            model.AddElement(x[pi], z, accept == 1)

    for u, v in G.edges:
        model.Add(x[u] != x[v])

    return model, x, y, z
