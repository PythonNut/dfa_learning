import torch
import random
from torch import nn
from torch.nn import functional as F
def logit(p):
    return (p/(1-p)).log()

class DFA(nn.Module):
    def __init__(self, n, s):
        super().__init__()
        self.n, self.s = n, s
        self.delta = nn.Parameter(torch.randn(s, n, n))
        self.f = nn.Parameter(logit(torch.rand(self.n)))

    def forward(self, s):
        q = torch.zeros(self.n)
        q[0] = 1

        delta = F.softmax(self.delta, dim=1)
        f = torch.sigmoid(self.f)
        for sym in s:
            q = delta[sym] @ q
        return f @ q

train = [
    ([1,1,1,1,1,1,1], 1),
    ([1,1,1,1,1,1], 0),
    ([1,1,1,1,1], 1),
    ([1,1,1,1], 0),
    ([1,1,1], 1),
    ([1,1],0),
    ([1], 1),

    ([1,1,1,1,0,1,1,1], 1),
    ([1,1,1,0,1,1,1], 0),
    ([1,1,0,0,0,0,1,1,0,1], 1),
    ([1,1,1,1], 0),
    ([1,1,1], 1),
    ([1,1],0),
    ([1,0,0,0,0], 1)
]

def main():
    n, s = 2, 2
    model = DFA(n, s)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(5000):
        random.shuffle(train)
        for x, y in train:
            model.zero_grad()
            y_pred = model(x)
            loss = - y*y_pred.log() - (1-y)*(1-y_pred).log()
            loss.backward()
            optim.step()
            print(model.f.data.tolist())
