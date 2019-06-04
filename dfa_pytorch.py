import torch
import random
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

    def forward(self, s):
        q = torch.zeros(self.n)
        q[0] = 1
        for sym in s:
            q = self.delta[sym] @ q
        return self.f @ q

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
            reg1 = model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1)
            reg_loss1 = (reg1.sum(1) - 1).pow(2).sum()
            reg2 = torch.cat((reg1, model.f.unsqueeze(0)))
            reg_loss2 = reg2.clamp(max=0).pow(2).sum() + (reg2-1).clamp(min=0).pow(2).sum()
            loss = - y*y_pred.log() - (1-y)*(1-y_pred).log() + 100 * reg_loss1 + 100 * reg_loss2
            loss.backward()
            optim.step()
            print(model.f.data.tolist())
