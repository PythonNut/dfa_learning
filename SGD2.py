import torch
from torch import nn
from torch.nn import functional as F
import random
class DFA(nn.Module):
	def __init__(self, n, s):
		super().__init__()
		self.n, self.s = n, s
		self.delta = nn.Parameter(torch.rand(s, n, n))
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
		for sym in s:
			q = self.delta[sym] @ q
		return q.dot(self.f)

class DFA2(nn.Module):
	def __init__(self, n, s,dele):
		super().__init__()
		self.n, self.s = n, s
		self.delta = dele
		self.f = nn.Parameter(torch.rand(self.n))
		self.q=nn.Parameter(torch.rand(self.n))
		self.normalize()

	def normalize(self):
		self.q.requires_grad = False
		self.q /= self.q.sum(0)
		#self.delta[:,0,:]=0
		self.q.requires_grad = True
	
	def forward(self, s):
		q = torch.zeros(self.n)
		q[0] = 1
		for sym in s:
			q = self.delta[sym] @ q
		return q.dot(self.f)
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
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
	random.shuffle(train)
	for x, y in train:
		model.zero_grad()
		y_pred = model(x)
		reg1 = model.delta.permute((0, 2, 1)).contiguous().view(n * s, -1)
		reg_loss1 = (reg1.sum(1) - 1).pow(2).sum()
		reg2 = torch.cat((reg1, model.f.unsqueeze(0)))
		reg_loss2 = reg2.clamp(max=0).pow(2).sum() + (reg2-1).clamp(min=0).pow(2).sum()
		reg_loss3 = model.delta.sum(0).sum(1)[0].pow(2)
		reg_loss4 = torch.reshape(torch.where(model.delta>1,model.delta,torch.zeros(s,n,n)),(-1,)).sum(0)
		loss_main =  y*y_pred.log() + (1-y)*(1-y_pred).log()
		loss = -loss_main + 10 * reg_loss1 + 20 * reg_loss2 + reg_loss3*10 +reg_loss4 * 10
		loss.backward()
		optim.step()
		model.normalize()
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
		print(model.delta)
		print("#####################")
		print(model.delta.grad)

model2 = DFA2(n,s,model.delta.data)
for epoch in range(1000):
	random.shuffle(train)
	for x, y in train:
		model2.zero_grad()
		y_pred = model2(x)
		reg_loss1 = torch.reshape(torch.where(model2.q<0,model2.q**2,torch.zeros(n)),(-1,)).sum(0)
		reg_loss2 = torch.reshape(torch.where(model2.q>1,model2.q,torch.zeros(n)),(-1,)).sum(0)
		reg_loss3 = torch.reshape(torch.where(model2.f<0,model2.f**2,torch.zeros(n)),(-1,)).sum(0)
		reg_loss4 = torch.reshape(torch.where(model2.f>1,model2.f,torch.zeros(n)),(-1,)).sum(0)
		loss_main =  y*y_pred.log() + (1-y)*(1-y_pred).log()
		loss2 = -loss_main + 10 * reg_loss1 + 10 * reg_loss2 + reg_loss3*10 +reg_loss4 * 10
		loss2.backward()
		optim.step()
		model2.normalize()
		print("------------------")
		print(model2.f)
		print(model2.q)
