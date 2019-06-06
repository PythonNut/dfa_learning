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
'''
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
'''
sampleSet=set([])
numSample =100
while len(sampleSet)<numSample:
    sample = tuple([ np.random.randint(2) for i in range(np.random.randint(100))])
    result = sample.count(1)%2
    sampleSet.add( (sample,result) )
sampleSet=list(sampleSet)
train = sampleSet[:int(numSample//4*3)]
test = sampleSet[int(numSample//4*3):]
#test=set(test)

loss_track=[]
def main():
    n, s = 2, 2
    model = DFA(n, s)
    optim = torch.optim.SGD(model.parameters(), lr=10)

    for epoch in range(10):
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
        sum +=int(np.absolute(y_pred.item() - y)<0.1)
        print("string "+str(x))
        print("compare actual   "+str(y) +" predict  "+str(y_pred))
        print("sum  "+str(sum))
    print(sum/len(test))
    return [(sum/len(test)),F.softmax(model.delta,dim=1),model.f]
def graph():
    plt.plot(loss_track)
    plt.title("average loss per epoch")
    plt.show()

def testing(num):
	score =0
	delta =[]
	final =[]
	record=[]
	for i in range(num):
		r,d,f = main()
		score+=r
		delta.append(d)
		final.append(f)
		record.append([r,d,f])
	print("final record over "+str(num)+" runs"+ str(score/num))
# observation 0.44 and 1 gives us 0.72. 0.42 occcurs when we have[[[1,1],[0,0]],[[1,0],[0,1]]
	return [[score,delta,final],record]
