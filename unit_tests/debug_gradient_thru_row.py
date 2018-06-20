import torch
from torch.autograd import Variable
num_units = 10
num_input = 100
dim = 11

X = Variable(torch.randn(num_input,dim),requires_grad=False)
W = Variable(torch.randn(num_units,dim),requires_grad=True)

W_out = Variable(torch.randn(num_units),requires_grad=True)

row_sum = W.sum(dim=0)
column_sum = W.sum(dim=1)
print(row_sum.shape)
print(column_sum.shape)

outy = W.mm(X.t())

print(outy.shape)


yhat = W_out.matmul(outy)

print(yhat.shape)

y = Variable(torch.randn(num_input),requires_grad=False)

output = (y-yhat).sum() + row_sum.sum() + column_sum.sum()

output.backward()

print(output)

print(W.grad)
print(W_out.grad)

