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
        # beta[(dim):(2dim)] = log(lam)
        # beta[(2dim):(3dim)] = log(eta)
        # beta[3dim] = log(sigma)
        # beta[3dim+1] = log(tau)

        return()

    def log_likelihood(self):
        w = self.beta[:self.dim]

        sigma = torch.exp(self.beta[3 * self.dim + 1])
        outy = (self.y - (self.X.mv(w))) * (self.y - (self.X.mv(w))) / (sigma * sigma) * 0.5
        out_sigma = torch.exp(self.beta[5*self.dim+2])
        return(outy)

    def log_prior(self):
        w = self.beta[:self.dim]
        r1_lam = self.beta[(self.dim):(2 * self.dim)]
        r2_lam = torch.exp(self.beta[(2 * self.dim):(3 * self.dim)])
        r1_eta = self.beta[(3 * self.dim):(4 * self.dim)]
        r2_eta = torch.exp(self.beta[(4 * self.dim):(5 * self.dim)])
        r1_tau = self.beta[5 * self.dim]
        r2_tau = torch.exp(self.beta[5 * self.dim + 1])
        lam = r1_lam * torch.sqrt(r2_lam)
        tau = r1_tau * torch.sqrt(r2_tau)
        eta = r1_eta * torch.sqrt(r2_eta)
        out_r1_lam = torch.dot(r1_lam, r1_lam) * 0.5
        out_r1_eta = torch.dot(r1_eta, r1_eta) * 0.5
        out_r1_tau = torch.dot(r1_tau, r1_tau) * 0.5
        out_r2_lam = ((self.nu*0.5+1)*torch.log(r2_lam) + self.nu*0.5/r2_lam).sum()
        out_r2_eta = ((self.nu*0.5+1)*torch.log(r2_eta) + self.nu*0.5/r2_eta).sum()
        out_r2_tau = ((self.nu*0.5+1)*torch.log(r2_tau) + self.nu*0.5/r2_tau).sum()

        out_w = (w * w/(tau*tau*lam*lam*eta*eta)).sum() * 0.5

        out_hessian = self.beta[(2 * self.dim):(3 * self.dim)].sum() + self.beta[(4 * self.dim):(5 * self.dim)].sum()+ self.beta[(4 * self.dim):(5 * self.dim)].sum() + self.beta[5 * self.dim + 1]

        out = out_r1_lam + out_r1_eta + out_r1_tau + out_r2_lam + out_r2_eta + out_r2_tau + out_w +out_hessian
        return(out)

    def forward(self):

        out_likelihoood = self.log_likelihood()
        out_prior = self.log_prior()
        out = -out_likelihoood - out_prior
        return(out)


    def load_explcit_gradient(self):
        return()

    def load_explicit_H(self):
        # write down explicit hessian
        return()
    def load_explcit_dH(self):
        # write down explicit 3 rd derivatives
        return()