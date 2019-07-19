import operator as op
from functools import reduce

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

def add_equals_1(M, V, lits):
    if isinstance(M, Minicard):
        M.add_atmost(lits, 1)
        M.add_clause(lits)
        return
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
