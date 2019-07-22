from boolexpr import *
from util import *
import itertools as it
import numpy as np
ctx = Context()

n = 5
m = 8
k = 8

x = ctx.get_vars('x', (0, k), (0, m), (0, 3))
y = ctx.get_vars('y', (0, m))

def sim_var(x, y, i, j):
    return or_s(and_s(x[i, j, 0], y[j]), and_s(x[i, j, 1], not_(y[j])))

def sim_clause(x, y, i):
    m = x.shape[1][1]
    return or_s(*(sim_var(x, y, i, j) for j in range(m)))

def sim_onehot(x):
    k = x.shape[0][1]
    m = x.shape[1][1]
    return and_s(*(onehot(*x[i, j, :]) for i, j in it.product(range(k), range(m))))

def sim_sat(x, y):
    k = x.shape[0][1]
    return and_s(*(sim_clause(x, y, i) for i in range(k)), sim_onehot(x))

def build_binomial(y, n):
    return and_s(*(or_(~y[i], ~y[j]) for i, j in it.combinations(range(n), 2)))

def satisfiability(a, vs):
    for v in vs:
        a = exists(v, a)
    return a

def universal_equality(a, b, vs):
    result = eq(a, b)
    for v in vs:
        result = forall(v, result)
    return result.simplify()

def build_problem(x, y, n):
   return and_s(
       universal_equality(build_binomial(y, n), satisfiability(sim_sat(x, y), y[n:]), y[:n]),
       break_symmetry_lex(x)
    )

def read_solution(ass, k, m):
    X = np.zeros((k, m, 3), np.bool)
    for var, val in ass.items():
        index = tuple(map(int, str(var)[2:-1].split(',')))
        X[index] = int(val)
    return X

def format_solution(X):
    result = []
    for clause in X:
        sub = []
        for i, var in enumerate(clause):
            if var[0]:
                sub.append(f'y{i}')
            elif var[1]:
                sub.append(f'~y{i}')
        result.append(f'[{", ".join(sub)}]')
    return ', '.join(result)

def format_solution2(X, y):
    result = []
    for clause in X:
        sub = []
        for i, var in enumerate(clause):
            if var[0]:
                sub.append(y[i])
            elif var[1]:
                sub.append(~y[i])
        result.append(or_(*sub))
    return and_(*result)

def digit_lt(d1, d2):
    return or_(and_(d1[1], d2[2]), and_(d1[0], or_(d2[1], d2[2])))

def digit_eq(d1, d2):
    return and_(eq(d1[0], d2[0]), eq(d1[1], d2[1]), eq(d1[2], d2[2]))

def number_lt(n1, n2, i=0):
    assert len(n1) == len(n2)
    if i + 1 == len(n1) == len(n2):
        return digit_lt(n1[i], n2[i])

    use_it = digit_lt(n1[i], n2[i])
    lose_it = and_(digit_eq(n1[i], n2[i]), number_lt(n1, n2, i+1))
    return or_(use_it, lose_it)

def break_symmetry_lex(x):
    k = x.shape[0][1]
    return and_s(*(number_lt(x[i], x[i+1]) for i in range(k-1)))

def main():
    problem = build_problem(x, y, n)
    status, sol = problem.sat()
    if not status:
        print("UNSAT")
    else:
        print(format_solution2(read_solution(sol, k, m), y))
