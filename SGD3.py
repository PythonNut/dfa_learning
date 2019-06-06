import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from queue import *
import random
def logit(p):
	return (p/(1-p)).log()
class DFA(nn.Module):
	def __init__(self, n, s):
		super().__init__()
		self.n, self.s = n, s
		self.delta = nn.Parameter(torch.rand(s, n, n))
		self.f = nn.Parameter(logit(torch.rand(n)))
		
		self.normalize()

	def normalize(self):
		self.delta.requires_grad = False
		self.delta /= self.delta.sum(1).unsqueeze(1).expand(self.delta.shape)
		#self.delta[:,0,:]=0
		self.delta.requires_grad = True
	
	def forward(self, s):
		q = torch.zeros(self.n)
		q[0] = 1
		delta = F.softmax(self.delta,dim=1)
		f =torch.sigmoid(self.f)
		for sym in s:
			q = pt(delta[sym],q)
			#q = delta[sym]@q
		return q.dot(self.f)

def mag(x):
	return x**2/sum(x)
def pt(x,y):
	x=x**5/sum(x)
	y=y**5/sum(y)
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
				if stuck>10:
					momentum = 10
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
optim = SGD(model.parameters(), lr=0.01)
stuck = 0
history = Queue(10)
past = -1
for _ in range(10):
	history.put(0)
for epoch in range(1000):
	random.shuffle(train)
	for x, y in train:
		model.zero_grad()
		y_pred = model(x)
		#reg1 = model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1)
		#reg_loss1 = (reg1.sum(1) - 1).pow(2).sum()
		#reg2 = torch.cat((reg1, model.f.unsqueeze(0)))
		#reg_loss2 = reg2.clamp(max=0).pow(2).sum() + (reg2-1).clamp(min=0).pow(2).sum()
		#reg_loss3 = model.delta.sum(0).sum(1)[0].pow(2)
		#reg_loss4 = torch.reshape(torch.where(model.delta>1,model.delta,torch.zeros(s,n,n)),(-1,)).sum(0)
		loss_main =  y*y_pred.log() + (1-y)*(1-y_pred).log()
		loss = -loss_main 
		if past == -1:
			past = loss
		else:
			past = (loss.data-past)**2
		first = torch.reshape(torch.where(model.delta>0.9,torch.zeros(s,n,n),model.delta),(-1,))
		diff = torch.reshape(torch.where(model.delta<0.1,torch.zeros(s,n,n),model.delta*5),(-1,)).sum()
		if past<0.5 and diff<0.5:
			stuck+=1
		#+ 10 * reg_loss1 + 10 * reg_loss2 + reg_loss3*10 +reg_loss4 * 10
		loss.backward()
		optim.step()
		if stuck>20:
			print("Stuck@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		print(stuck)
		print(loss)
		print("#############################")
		print(F.softmax(model.delta))
		print("---------------")
		print(model.delta.grad)


