import random
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

        delta = self.delta.softmax(dim=1)
        f = self.f.sigmoid()
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
    ([], 0),

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
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        random.shuffle(train)
        for x, y in train:
            model.zero_grad()
            y_pred = model(x)
            loss = - y*y_pred.log() - (1-y)*(1-y_pred).log()
            loss.backward()
            optim.step()
            print(model.f)

def main2(pos,neg):
	n, s = 7, 2
	model = DFA(n, s)
	optim = torch.optim.Adam(model.parameters(), lr=0.001)

	#pos = [(list(map(int, str(bin(7 * i))[2:])), 1) for i in range(1000)]
	#neg = [(list(map(int, str(bin(7 * i + random.randint(1, 6)))[2:])), 0) for i in range(1000)]

	random.shuffle(pos)
	random.shuffle(neg)

	train_pos, test_pos = pos[:700], pos[700:]
	train_neg, test_neg = neg[:700], neg[700:]

	train = train_pos + train_neg
	for epoch in range(5000):
    	random.shuffle(train)
    	for x, y in train:
        	model.zero_grad()
        	y_pred = model(x)
        	reg = torch.cat((model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1), model.f.unsqueeze(0)))
        	reg_loss1 = (reg.sum(1) - 1).pow(2).sum()
       		reg_loss2 = reg.clamp(max=0).abs().pow(2).sum() + (reg-1).clamp(min=0).pow(2).sum()
       		loss = - y*y_pred.log() - (1-y)*(1-y_pred).log() + 100 * reg_loss1 + 100 * reg_loss2
        	loss.backward()
        	optim.step()

    	acc = 0
    	for x, y in test_pos+test_neg:
        	print(model(x))
        	acc += (model(x).item() > 0) == (y > 0)

    	print(acc/len(test_pos + test_neg))
