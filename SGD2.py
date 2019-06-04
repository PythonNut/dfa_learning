import torch
from torch import nn
from torch.nn import functional as F
import random
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
		self.delta[:,0,:]=0
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
	([1], 1),
	([1,1,1,1,0,1,1,1], 1),
	([1,1,1,0,1,1,1], 0),
	([1,1,0,0,0,0,1,1,0,1], 1),
	([1,1,1,1], 0),
	([1,1,1], 1),
	([1,1],0),
	([1,0,0,0,0], 1)
]


n,s=3,2
model = DFA(n, s)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
	random.shuffle(train)
	for x, y in train:
		model.zero_grad()
		y_pred = model(x)
		reg = torch.cat((model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1), model.f.unsqueeze(0)))
		reg_loss1 = (reg.sum(1) - 1).pow(2).sum()
		reg_loss2 = reg.clamp(max=0).pow(2).sum() + (reg-1).clamp(min=0).pow(2).sum()
		reg_loss3 = model.delta.sum(0).sum(1)[0].pow(2)
		reg_loss4 = 0  #torch.reshape(torch.where(model.delta<0,-1*model.delta,torch.zeros(s,n,n)),(-1,)).sum(0)
		loss = - y*y_pred.log() - (1-y)*(1-y_pred).log() + 100 * reg_loss1 + 100 * reg_loss2 + reg_loss3*100 +reg_loss4 * 100
		loss.backward()
		optim.step()
		model.normalize()
		print(model.delta)
		
