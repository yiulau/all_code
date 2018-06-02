from explicit.genleapfrog_ult_util import eigen
import torch,numpy
inp = torch.eye(5)
input = numpy.array([[0.0,1],[-1,0]])
#print(input)

#a = torch.from_numpy(input)
#print(a)
#lam,Q = a.eig(eigenvectors=True)
#print(lam)
#exit()
#lam[:,1]==0
#exit()
#print(inp)

lam,Q = eigen(inp)

exit()

print(lam)
print(Q)

print(Q.mm((lam)).mm(Q.t()))