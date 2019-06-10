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
def data():
	fileTrain = open("trainA.txt","r")
	#fileRealTest = open("testA.txt","r")

	data = fileTrain.readlines()
	index = data[0].find(" ")
	if index>-1:
		num = int(data[0][:index])
		s = int(data[0][index+1:])
	data = data[1:]
	#realTest = fileRealTest.readlines()
	process=[]
	for j in range(num):
		i =  data[j]
		y=int(i[0])
		i=i[2:]
		index = i.find(" ")
		i = i[index+1:]
		i= i[:-1]
		x=list(map(int,i.split()))
		process.append([x,y])
	random.shuffle(process)
	train = process[:(len(process)//4*3)]
	test = process[len(process)//4*3:]
	return [1000,s,train,test]
def main():
	n,s, train,test=data()
	model = DFA(n, s)
	optim = torch.optim.Adam(model.parameters(), lr=.1, weight_decay=1e-2)
	eps = 1e-12
	counter = 0 
	#wrong = True
	for epoch in range(10):
		random.shuffle(train)
		for x,y in train:
			model.zero_grad()
			y_pred = model(x)
			loss = - y*(y_pred+eps).log() - (1-y)*(1-y_pred+eps).log()
			loss.backward()
			optim.step()
			print("training: "+str(counter/10/len(train)))
			counter+=1
	sum = 0
	for x, y in train:
		y_pred = model(x)
		sum += int((y_pred.item() > 0.5) == (y > 0.5))
	train_score = (sum/len(train))
	sum = 0
	for x, y in test:
		y_pred = model(x)
		sum += int((y_pred.item() > 0.5) == (y > 0.5))
		#print("string "+str(x))
		#print("compare actual   "+str(y) +" predict  "+str(y_pred))
		#print("sum  "+str(sum))
	test_score = (sum/len(test))
	print((train_score, test_score))
	return [(train_score, test_score), F.softmax(model.delta, dim=1) > 0.9, model.f > 0.9]


def graph():
	plt.plot(loss_track)
	plt.title("average loss per epoch")
	plt.show()


def testing(num):
	score = 0
	scorels = []
	delta = []
	final = []
	record = []
	for i in range(num):
		print(str(i) + "  DFA")
		r, d, f = main()
		r0, r1 = r
		score += r1
		scorels.append(r)
		delta.append(d)
		final.append(f)
		record.append([r, d, f])
	print("final record over "+str(num)+" runs" + str(score/num))
# observation 0.44 and 1 gives us 0.72. 0.42 occcurs when we have[[[1,1],[0,0]],[[1,0],[0,1]]
	return [[score, scorels, delta, final], record]
