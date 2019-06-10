import torch
import random
from torch import nn
from torch.nn import functional as F
import numpy as np

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

def generator(num_samples, max_str_length):
    '''string generator for odd number of one dfa'''
    sampleSet=set([])
    while len(sampleSet)<num_samples:
        sample = tuple([np.random.randint(2) for i in range(np.random.randint(max_str_length))])
        result = sample.count(1)%2
        sampleSet.add((sample,result))
    sampleSet=list(sampleSet)
    random.shuffle(sampleSet)
    train = sampleSet[:int(num_samples//4*3)]
    test = sampleSet[int(num_samples//4*3):]
    unique_test = [value for value in test if value not in train]
    return train,test,unique_test
        

def main():
    train,test,unique_test=generator(500,100)
    n,s = 2,2
    model = DFA(n, s)
    optim = torch.optim.Adam(model.parameters(), lr=1, weight_decay=1e-2)
    eps = 1e-12
    counter = 0 
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
        y_pred = model(list(x))
        print("train ypred",y_pred,"train y",y)
        sum += int((y_pred.item() > 0.8) == (y > 0.8))
    train_score = (sum/len(train))
    
    sum = 0
    for x, y in unique_test:
        print(list(x))
        y_pred = model(list(x))
        print("ftest ypred",y_pred,"ftest y",y)
        sum += int((y_pred.item() > 0.8) == (y > 0.8))
    unique_test_score = (sum/len(unique_test))
    
    sum = 0
    for x,y in test:
        print(list(x))
        y_pred = model(list(x))
        print("test ypred",y_pred,"test y",y)
        sum += int((y_pred.item() > 0.8) == (y > 0.8))
    test_score = (sum/len(test))
        
    print((train_score, unique_test_score, test_score))
    return [(train_score, unique_test_score, test_score), F.softmax(model.delta, dim=1) > 0.9, model.f > 0.9]

def performance(runs):
    train_sum = 0
    unique_test_sum = 0
    test_sum = 0
    for i in range(runs):
        print(i)
        train_sum += main()[0][0]
        unique_test_sum += main()[0][1]
        test_sum += main()[0][2]
    train_sum /= runs
    unique_test_sum /= runs 
    test_sum /= runs
    return (train_sum,unique_test_sum,test_sum)