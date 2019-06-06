import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from queue import *
import random
'''
def sgd(rate=0.01,symLs,size,inputLs,outputLs, states):
	#states is number of states
	# size is number of samples
	symTen = dict({})
	for i in symLs:
		symTen[i] = torch.ones(states,states,requires_grad=True)/states/states
	final = torch.ones(states,1,requires_grad=True)/states
	for i in len(size):
		tran = torch.tensor(input)
		for j in inputL[i]:
			tran = torch.mm(tran,symTen[j])
		tran = torch.mm(tran,final)
		loss = torch.sum( (input - tran)**2 )
		#add regularizer later
		loss.backward()
		if i == size - 1:
			print("symTen: ",symTen)
			print("final: ",final)
		with torch.no_grad():
			for k in symLs:
				symTen[k] = symTen[k] - rate * symTen[k].grad
			final = final - rate * final.grad
		for k in symLs:
			symTen[k].requires_grad = True
		final.requires_grad = True

class DFA(nn.Module):
	def __init__(self,sym,states):
		super().__init__()
		self.ls = nn.Parameter(torch.ones((sym,states,states),dtype=torch.float64,requires_grad=True)/states/states)
			
		#self.final=nn.Parameter(torch.ones((states,1),dtype=torch.float64,requires_grad=True))
		
	def forward(self,input):
		tran = torch.tensor([1.0,0],dtype=torch.float64)
		
		for j in input:
			tran = tran @ self.ls[j]
		tran = tran @ self.ls[j]
		return tran
'''

class DFA(nn.Module):
	def __init__(self, n, s):
		super().__init__()
		self.n, self.s = n, s
		self.delta = nn.Parameter(torch.rand(s, n, n),requires_grad=True)
		self.f = torch.tensor([0.0,1.0])
		self.normalize()

	def normalize(self):
		self.delta.requires_grad = False
		self.delta /= self.delta.sum(1).unsqueeze(1).expand(self.delta.shape)
		#self.delta[:,0,:]=0
		self.delta.requires_grad = True
	
	def forward(self, s):
		q = torch.zeros(self.n)
		q[0] = 1
		delta = self.delta
		#delta = F.softmax(self.delta,dim=1)
		#f = F.sigmoid(self.f)
		for sym in s:
			q = delta[sym] @ q
		return q.dot(self.f)

def pt(x,y):
	x=x**2/sum(x)
	y=y**2/sum(y)
	product = x@y
	return product/sum(product)
a=np.array([[0.1,0.9],[0.9,0.1]])
b=np.array([0.1,0.9])
class SGD(Optimizer):
	def __init__(self, params, lr, momentum=0, dampening=0,
				 weight_decay=0, nesterov=False):

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(SGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(SGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				if weight_decay != 0:
					d_p.add_(weight_decay, p.data)
				if momentum != 0:
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				p.data.add_(-group['lr'], d_p)

		return loss

train = [
	([1], 1),
	([1,1],0),
	([1,0], 1),
	([0],0),
	([],0),
	([1,0,1],0),
	([1,1,1,1,1],1),
	([1,1,1,1,1,1,1,1],0),
	([1,1,1],1),
	([1,1,1,1],0),
	([1,1,1,1,1],1),
	([1,1,1,1,1,1],0),
	([0,0,1],1),
	([0,0,0,1],1),
	([0,0,0,0,1],1)
]
n,s=2,2
model = DFA(n, s)

stuck = 0
past = Queue(10)
for _ in range(10):
	past.put(0)
optim =SGD(model.parameters(),lr=0.001)
for epoch in range(1000):
	random.shuffle(train)
	for x, y in train:
		model.zero_grad()
		y_pred = model(x)
		#reg1 = model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1)
		loss_main =  y*y_pred.log() + (1-y)*(1-y_pred).log()
		loss = -loss_main
		loss.backward()
		optim.step()
		model.normalize()
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
		print(model.delta)
		print("#####################")
		print(model.delta.grad)