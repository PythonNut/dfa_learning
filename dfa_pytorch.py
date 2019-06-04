import torch
from torch import nn
from torch.nn import functional as F

class DFA(nn.Module):
    def __init__(self, n, s):
        super().__init__()
        self.n, self.s = n, s
        self.delta = nn.Parameter(torch.rand(s, n, n))
        self.f = nn.Parameter(torch.rand(self.n))
        self.normalize()

    def normalize(self):
        self.delta.requires_grad = False
        self.delta /= self.delta.sum(1).unsqueeze(1).expand(self.delta.shape)
        self.delta.requires_grad = True

        self.f.requires_grad = False
        self.f /= self.f.sum()
        self.f.requires_grad = True

    def forward(self, s):
        q = torch.zeros(self.n)
        q[0] = 1
        for sym in s:
            q = self.delta[sym] @ q
        return q.dot(self.f)

train = [
    ([1,1,1,1,1,1,1], 1),
    ([1,1,1,1,1,1], 0),
    ([1,1,1,1,1], 1),
    ([1,1,1,1], 0),
    ([1,1,1], 1),
    ([1,1],0),
    ([1], 1)
]

model = DFA(2, 2)
optim = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    for x, y in train:
        model.zero_grad()
        y_pred = model(x)
        loss = - y*y_pred.log() - (1-y)*(1-y_pred).log()
        loss.backward()
        optim.step()
        model.normalize()
        print(model.delta)
