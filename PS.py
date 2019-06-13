from pyswarm import pso
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

def question(delta,ls,y_p):
	q = [1,0]
	f =[0,1]
	for p in ls:
		q = delta[p]@q
	y=f@q
	loss =  -y*np.log(y_p) - y_p*np.log(y)
	return loss

def lossFunction(delta):
	loss = 0
	for ls,y_p in train:
		loss += question(delta, ls, y_p)
	return np.sum(loss)

def constrain(delta):
	con =delta.reshape(1,-1)
	cont = np.where(con<=0,-1,con)
	cont = np.where(cont >1, -1, con)
	bound =[]
	for i in range(len(delta)):
		bound
	return cont

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
loss_track=[]
def main():
	n, s = 2, 2
	model = DFA(n, s)
	optim = torch.optim.SGD(model.parameters(), lr=10)

	for epoch in range(10000):
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

def graph():
	plt.plot(loss_track)
	plt.title("average loss per epoch")
	plt.show()
