import torch
from torch.autograd import Variable
num_units = 5
num_input = 100
dim = 11


Y = Variable(torch.randn(num_units,2),requires_grad=True)

Y_1 = Y[:,0]
Y_2 = Y[:,1]

Y_re = torch.cat([Y_1,Y_2],dim=0)

out = (Y_re * Y_re).sum() * 0.5

out.backward()
print(Y.grad)

print(Y)

exit()
X_1 = Variable(torch.randn(num_units,1),requires_grad=True)
X_2 = Variable(torch.randn(num_units,1),requires_grad=True)

X_3 = torch.cat([X_1,X_2],dim=0)

out = (X_3*X_3).sum() * 0.5

out.backward()

print(X_1.grad)
print(X_1)
print(X_2.grad)
print(X_2)
exit()


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

