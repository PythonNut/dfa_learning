import torch
import torch.nn as nn
import torch.nn.functional as F
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
'''
class Model(nn.Module):
	def __init__(self,sym,states):
		super().__init__()
		self.ls = nn.Parameter(torch.ones((sym,states,states),dtype=torch.float64,requires_grad=True)/states/states)
			
		self.final=nn.Parameter(torch.ones((states,1),dtype=torch.float64,requires_grad=True))

		
	def forward(self,input):
		tran = torch.tensor([1.0,0],dtype=torch.float64)
		for j in input:
			tran = tran @ self.ls[j]
		tran = tran @ self.ls[j]
		return tran
class SGD(): 
	def sgd2(self,sym,data,states): 
		learning_rate = 1e-4
		model = Model(sym,states)
		loss_fn = F.binary_cross_entropy
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
		for input,output in data:
			y_pred = model(input)
			loss = loss_fn(y_pred, output)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print(model)
		
SGD().sgd2(2,[[[1,1,1],1],[[1],1]],2)
