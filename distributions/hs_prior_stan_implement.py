from abstract.abstract_class_V import V
import torch
import torch.nn as nn


class V_test_abstract(V):
    def __init__(self):
        super(V_test_abstract, self).__init__()

    def V_setup(self,y,X,nu):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.dim = X.shape[1]
        self.beta = nn.Parameter(torch.zeros(self.dim*3+2),requires_grad=True)
        self.y = y
        self.X = X
        self.nu = nu

        # beta[:dim] = w
        # beta[(dim):(2dim)] = r1
        # beta[(2dim):(3dim)] = log(r2)
        # beta[2dim] = log(tau)
        # beta[2dim+1] = log(sigma)

        return()

    def forward(self):
        w = self.beta[:self.dim]
        r1 = self.beta[self.dim:2*self.dim]
        r2 = torch.exp(self.beta[(2*self.dim):(3*self.dim)])
        lam = r1 * torch.sqrt(r2)
        r1_tau = self.beta[3 * self.dim]
        r2_tau = torch.exp(self.beta[3 * self.dim+1])
        tau = r1_tau * torch.sqrt(r2_tau)
        sigma = torch.exp(self.beta[3*self.dim+2])


        outy = (self.y - (self.X.mv(w)))*(self.y - (self.X.mv(w)))/(sigma*sigma) * 0.5
        outw = (w * w /(tau*tau*lam*lam)).sum() * 0.5
        outr1 = torch.dot(r1,r1)
        outr2 = ((self.nu*0.5+1)*torch.log(r2) + self.nu*0.5/r2).sum()
        outr1_tau = torch.dot(r1_tau,r1_tau)
        outr2_tau = ((self.nu*0.5+1)*torch.log(r2_tau) + self.nu*0.5/r2_tau)
        out_hessian = self.beta[(2*self.dim):(3 * self.dim)].sum() + self.beta[3*self.dim] + self.beta[3*self.dim+1] +self.beta[3*self.dim+2]
        out = outy + outw + outr1 + outr2+ outr1_tau + outr2_tau + out_hessian
        return(out)

    def load_explcit_gradient(self):
        return()

    def load_explicit_H(self):
        # write down explicit hessian
        return()
    def load_explcit_dH(self):
        # write down explicit 3 rd derivatives
        return()