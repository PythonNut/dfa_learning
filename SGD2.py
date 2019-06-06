import matplotlib.pyplot as plt
import torch
import numpy as np
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

    ([1,1,1,1,0,1,1,1], 1),
    ([1,1,1,0,1,1,1], 0),
    ([1,1,0,0,0,0,1,1,0,1], 1),
    ([1,1,1,1], 0),
    ([1,1,1], 1),
    ([1,1],0),
    ([1,0,0,0,0], 1)
]
test=[]
for i in range(np.random.randint(100)):
    sample = [ np.random.randint(2) for i in range(np.random.randint(100))]
    result = np.sum(sample)%2
    test.append( (sample,result) )
#test=set(test)

loss_track=[]
def main():
    n, s = 2, 2
    model = DFA(n, s)
    optim = torch.optim.SGD(model.parameters(), lr=10)

    for epoch in range(100):
        random.shuffle(train)
        lossRecord =[]
        for x, y in train:     
            model.zero_grad()
            y_pred = model(x)
            loss = - y*y_pred.log() - (1-y)*(1-y_pred).log()
            loss.backward()
            optim.step()
            print(F.softmax(model.delta,dim=1))
            print(model.f.data.tolist())
            lossRecord.append(loss.item())
        loss_track.append(np.mean(lossRecord))
    sum =0
    for x,y in test:
        y_pred = model(x)
        sum +=int(np.absolute(y_pred.item() - y)<0.01)
        print("compare actual   "+str(y) +" predict  "+str(y_pred))
        print("sum  "+str(sum))
    print(sum/len(test))
def graph():
    plt.plot(loss_track)
    plt.title("average loss per epoch")
    plt.show()
