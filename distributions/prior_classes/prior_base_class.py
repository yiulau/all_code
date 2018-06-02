import abc, torch
import torch.nn as nn

class prior_class(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,obj):
        self.V = obj
    @abc.abstractmethod
    def create_hyper_par_fun(self):
        pass

    @abc.abstractmethod
    def prior_forward(self):
        pass
# def generate_prior_dict():
#     def create_hyper_par_fun(obj):



def hs_prior_stan(list_z,list_r1_local,list_log_r2_local,list_r1_global,list_log_r2_global,nu):
    assert len(list_z)==len(list_r1_local)==len(list_log_r2_local)==len(list_r1_global)==len(list_log_r2_global)
    out = 0
    for i in range(len(list_z)):
        z = list_z[i]
        r1_local = list_r1_local[i]
        r2_local = torch.exp(list_log_r2_local[i])
        r1_global = list_r1_global[i]
        r2_global = torch.exp(list_log_r2_global[i])
        lam = r1_local * torch.sqrt(r2_local)
        tau = r1_global * torch.sqrt(r2_global)
        outz = -(z * z ).sum() * 0.5
        outr1_local = - (r1_local*r1_local).sum() * 0.5
        outr2_local = -((nu*0.5+1)*torch.log(r2_local) + nu*0.5/r2_local).sum()
        outr1_global = - (r1_global*r1_global).sum() * 0.5
        outr2_global = -((nu * 0.5 + 1) * torch.log(r2_global) + nu * 0.5 / r2_global).sum()
        outhessian = - list_log_r2_local[i].sum() - list_log_r2_global[i].sum()
        out += outz + outr1_local + outr2_local +outr1_global + outr2_global+ outhessian

    return(out)

def hs_prior_stan_cp(list_w,list_r1_local,list_log_r2_local,list_r1_global,list_log_r2_global,nu):
    assert len(list_w)==len(list_r1_local)==len(list_log_r2_local)==len(list_r1_global)==len(list_log_r2_global)
    out = 0
    for i in range(len(list_w)):
        w = list_w[i]
        r1_local = list_r1_local[i]
        r2_local = torch.exp(list_log_r2_local[i])
        r1_global = list_r1_global[i]
        r2_global = torch.exp(list_log_r2_global[i])
        lam = r1_local * torch.sqrt(r2_local)
        tau = r1_global * torch.sqrt(r2_global)
        outw = -(w * w / (tau * tau * lam * lam )).sum() * 0.5
        outr1_local = - (r1_local*r1_local).sum() * 0.5
        outr2_local = -((nu*0.5+1)*torch.log(r2_local) + nu*0.5/r2_local).sum()
        outr1_global = - (r1_global*r1_global).sum() * 0.5
        outr2_global = -((nu * 0.5 + 1) * torch.log(r2_global) + nu * 0.5 / r2_global).sum()
        outhessian = - list_log_r2_local[i].sum() - list_log_r2_global[i].sum()
        out += outw + outr1_local + outr2_local +outr1_global + outr2_global+ outhessian

    return(out)
def hs_prior_stan_prepare(obj):
    # self.target_param_dict = nn.Module.named_parameters()
    assert hasattr(obj,"list_target_param_shape_dict")
    setattr(obj,name="list_z",value=[])
    setattr(obj,name="list_r1_local",value=[])
    setattr(obj,name="list_log_r2_local",value=[])
    setattr(obj, name="list_r1_global", value=[])
    setattr(obj, name="list_log_r2_global", value=[])
    for name,this_shape in obj.list_target_param_shape_dict:
        z = nn.Parameter(torch.zeros(this_shape))
        r1_local = nn.Parameter(torch.zeros(this_shape),requires_grad=True)
        log_r2_local = nn.Parameter(torch.zeros(this_shape),requires_grad=True)
        r1_global = nn.Parameter(torch.zeros(1),requires_grad=True)
        log_r2_global = nn.Parameter(torch.zeros(1),requires_grad=True)
        setattr(obj, name=name + "_z", value=z)
        setattr(obj,name=name+"_r1_local",value=r1_local)
        setattr(obj,name=name+"_log_r2_local",value=log_r2_local)
        setattr(obj, name=name + "_r1_global", value=r1_global)
        setattr(obj, name=name + "_log_r2_global", value=log_r2_global)
        obj.list_z.append((name+"_z",z))
        obj.list_r1_local.append((name+"_r1_local",r1_local))
        obj.list_log_r2_local.append((name+"_log_r2_local",log_r2_local))
        obj.list_r1_global.append((name + "_r1_local", r1_global))
        obj.list_log_r2_global.append((name + "_log_r2_local", log_r2_global))
    return()

def hs_prior_stan_cp_prepare(obj):
    # self.target_param_dict = nn.Module.named_parameters()
    assert hasattr(obj,"list_target_param_shape_dict")
    setattr(obj,name="list_w",value=[])
    setattr(obj,name="list_r1_local",value=[])
    setattr(obj,name="list_log_r2_local",value=[])
    setattr(obj, name="list_r1_global", value=[])
    setattr(obj, name="list_log_r2_global", value=[])
    for name,this_shape in obj.list_target_param_shape_dict:
        w = nn.Parameter(torch.zeros(this_shape))
        r1_local = nn.Parameter(torch.zeros(this_shape),requires_grad=True)
        log_r2_local = nn.Parameter(torch.zeros(this_shape),requires_grad=True)
        r1_global = nn.Parameter(torch.zeros(1),requires_grad=True)
        log_r2_global = nn.Parameter(torch.zeros(1),requires_grad=True)
        setattr(obj, name=name + "_w", value=w)
        setattr(obj,name=name+"_r1_local",value=r1_local)
        setattr(obj,name=name+"_log_r2_local",value=log_r2_local)
        setattr(obj, name=name + "_r1_global", value=r1_global)
        setattr(obj, name=name + "_log_r2_global", value=log_r2_global)
        obj.list_w.append((name+"_w",w))
        obj.list_r1_local.append((name+"_r1_local",r1_local))
        obj.list_log_r2_local.append((name+"_log_r2_local",log_r2_local))
        obj.list_r1_global.append((name + "_r1_local", r1_global))
        obj.list_log_r2_global.append((name + "_log_r2_local", log_r2_global))
    return()

def hs_prior_raw(list_w,list_target_log_lam,list_target_log_tau,nu):
    assert len(list_w) == len(list_target_log_lam)== len(list_target_log_tau)
    out =0
    for i in range(len(list_w)):
        w = list_w[i]
        lam = torch.exp(list_target_log_lam[i])
        tau = torch.exp(list_target_log_tau[i])
        outw = -(w * w / (lam * lam * tau * tau )).sum() * 0.5
        outlam = -((nu +1. )*0.5 + torch.log(1+ (1/nu)* (lam*lam))).sum()
        outtau = -((1 +1. )*0.5 + torch.log(1+ (1/1)* (lam*lam))).sum()
        outhessian = - list_target_log_lam[i].sum() - list_target_log_tau[i].sum()
        out += outw + outlam + outtau + outhessian

    return(out)

def hs_prior_raw_ncp(list_z,list_target_log_lam,list_target_log_tau,nu):
    assert len(list_z) == len(list_target_log_lam)== len(list_target_log_tau)
    out =0
    for i in range(len(list_z)):
        z = list_z[i]
        lam = torch.exp(list_target_log_lam[i])
        tau = torch.exp(list_target_log_tau[i])
        outz = -(z*z).sum()*0.5
        outlam = -((nu +1. )*0.5 + torch.log(1+ (1/nu)* (lam*lam))).sum()
        outtau = -((1 +1. )*0.5 + torch.log(1+ (1/1)* (lam*lam))).sum()
        outhessian = - list_target_log_lam[i].sum() - list_target_log_tau[i].sum()
        out += outz + outlam + outtau + outhessian

    return(out)
def hs_prior_raw_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_w", value=[])
    setattr(obj, name="list_log_lam", value=[])
    setattr(obj, name="list_log_tau", value=[])
    for name, this_shape in obj.list_target_param_shape_dict:
        w = nn.Parameter(torch.zeros(this_shape))
        log_lam = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_w", value=w)
        setattr(obj, name=name + "_log_lam", value=log_lam)
        setattr(obj, name=name + "_log_tau", value=log_tau)
        obj.list_w.append((name + "_w", w))
        obj.list_log_lam.append((name + "_log_lam", log_lam))
        obj.list_log_tau.append((name + "_log_tau", log_tau))

    return()

def hs_prior_raw_ncp_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_z", value=[])
    setattr(obj, name="list_log_lam", value=[])
    setattr(obj, name="list_log_tau", value=[])
    for name, this_shape in obj.list_target_param_shape_dict:
        z = nn.Parameter(torch.zeros(this_shape))
        log_lam = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_z", value=z)
        setattr(obj, name=name + "_log_lam", value=log_lam)
        setattr(obj, name=name + "_log_tau", value=log_tau)
        obj.list_w.append((name + "_z", z))
        obj.list_log_lam.append((name + "_log_lam", log_lam))
        obj.list_log_tau.append((name + "_log_tau", log_tau))

    return()


def hs_plus_stan(list_z,list_r1_local,list_log_r2_local,list_r1_local_plus,list_log_r2_local_plus,
                 list_r1_global,list_log_r2_global,nu):
    assert len(list_z)==len(list_r1_local)==len(list_log_r2_local)==len(list_r1_local_plus)==\
           len(list_log_r2_local_plus)==len(list_r1_global)==len(list_log_r2_global)
    out = 0
    for i in range(len(list_z)):
        z = list_z[i]
        r1_local = list_r1_local[i]
        r2_local = torch.exp(list_log_r2_local[i])
        r1_local_plus = list_r1_local_plus[i]
        r2_local_plus = torch.exp(list_log_r2_local_plus[i])
        r1_global = list_r1_global[i]
        r2_global = torch.exp(list_log_r2_global[i])
        lam = r1_local * torch.sqrt(r2_local)
        lam_plus = r1_local_plus * r2_local_plus
        tau = r1_global * torch.sqrt(r2_global)
        outz = -(z * z ).sum() * 0.5
        outr1_local = - (r1_local *r1_local).sum() * 0.5
        outr2_local = -((nu*0.5+1)*torch.log(r2_local) + nu*0.5/r2_local).sum()
        outr1_local_plus = - (r1_local_plus*r1_local_plus).sum() * 0.5
        outr2_local_plus = -((nu*0.5+1)*torch.log(r2_local_plus) + nu*0.5/r2_local_plus).sum()
        outr1_global = - (r1_global* r1_global).sum() * 0.5
        outr2_global = -((nu * 0.5 + 1) * torch.log(r2_global) + nu * 0.5 / r2_global).sum()
        outhessian = - list_log_r2_local[i].sum() - list_log_r2_local_plus[i].sum() - list_log_r2_global[i].sum()
        out += outz + outr1_local + outr2_local +outr1_local_plus + outr2_local_plus+\
               outr1_global + outr2_global+ outhessian

    return(out)


def hs_plus_cp_stan(list_w,list_r1_local,list_log_r2_local,list_r1_local_plus,list_log_r2_local_plus,
                 list_r1_global,list_log_r2_global,nu):
    assert len(list_w)==len(list_r1_local)==len(list_log_r2_local)==len(list_r1_local_plus)==\
           len(list_log_r2_local_plus)==len(list_r1_global)==len(list_log_r2_global)
    out = 0
    for i in range(len(list_w)):
        w = list_w[i]
        r1_local = list_r1_local[i]
        r2_local = torch.exp(list_log_r2_local[i])
        r1_local_plus = list_r1_local_plus[i]
        r2_local_plus = torch.exp(list_log_r2_local_plus[i])
        r1_global = list_r1_global[i]
        r2_global = torch.exp(list_log_r2_global[i])
        lam = r1_local * torch.sqrt(r2_local)
        lam_plus = r1_local_plus * r2_local_plus
        tau = r1_global * torch.sqrt(r2_global)
        outw = -(w * w / (tau * tau * lam * lam * lam_plus * lam_plus)).sum() * 0.5
        outr1_local = - (r1_local*r1_local).sum() * 0.5
        outr2_local = -((nu*0.5+1)*torch.log(r2_local) + nu*0.5/r2_local).sum()
        outr1_local_plus = - (r1_local_plus*r1_local_plus).sum() * 0.5
        outr2_local_plus = -((nu*0.5+1)*torch.log(r2_local_plus) + nu*0.5/r2_local_plus).sum()
        outr1_global = - (r1_global*r1_global).sum() * 0.5
        outr2_global = -((nu * 0.5 + 1) * torch.log(r2_global) + nu * 0.5 / r2_global).sum()
        outhessian = - list_log_r2_local[i].sum() - list_log_r2_local_plus[i].sum() - list_log_r2_global[i].sum()
        out += outw + outr1_local + outr2_local +outr1_local_plus + outr2_local_plus+\
               outr1_global + outr2_global+ outhessian

    return(out)

def hs_plus_stan_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_z", value=[])
    setattr(obj, name="list_r1_local", value=[])
    setattr(obj, name="list_log_r2_local", value=[])
    setattr(obj, name="list_r1_local_plus", value=[])
    setattr(obj, name="list_log_r2_local_plus", value=[])
    setattr(obj, name="list_r1_global", value=[])
    setattr(obj, name="list_log_r2_global", value=[])

    for name, this_shape in obj.list_target_param_shape_dict:
        z = nn.Parameter(torch.zeros(this_shape))
        r1_local = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_r2_local = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        r1_local_plus = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_r2_local_plus = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        r1_global = nn.Parameter(torch.zeros(1), requires_grad=True)
        log_r2_global = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_z", value=z)
        setattr(obj, name=name + "_r1_local", value=r1_local)
        setattr(obj, name=name + "_log_r2_local", value=log_r2_local)
        setattr(obj, name=name + "_r1_local_plus", value=r1_local_plus)
        setattr(obj, name=name + "_log_r2_local_plus", value=log_r2_local_plus)
        setattr(obj, name=name + "_r1_global", value=r1_global)
        setattr(obj, name=name + "_log_r2_global", value=log_r2_global)
        obj.list_z.append((name + "_z", z))
        obj.list_r1_local.append((name + "_r1_local", r1_local))
        obj.list_log_r2_local.append((name + "_log_r2_local", log_r2_local))
        obj.list_r1_local_plus.append((name + "_r1_local_plus", r1_local_plus))
        obj.list_log_r2_local_plus.append((name + "_log_r2_local_plus", log_r2_local_plus))
        obj.list_r1_global.append((name + "_r1_local", r1_global))
        obj.list_log_r2_global.append((name + "_log_r2_local", log_r2_global))


def hs_plus_cp_stan_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_w", value=[])
    setattr(obj, name="list_r1_local", value=[])
    setattr(obj, name="list_log_r2_local", value=[])
    setattr(obj, name="list_r1_local_plus", value=[])
    setattr(obj, name="list_log_r2_local_plus", value=[])
    setattr(obj, name="list_r1_global", value=[])
    setattr(obj, name="list_log_r2_global", value=[])

    for name, this_shape in obj.list_target_param_shape_dict:
        w = nn.Parameter(torch.zeros(this_shape))
        r1_local = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_r2_local = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        r1_local_plus = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_r2_local_plus = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        r1_global = nn.Parameter(torch.zeros(1), requires_grad=True)
        log_r2_global = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_w", value=w)
        setattr(obj, name=name + "_r1_local", value=r1_local)
        setattr(obj, name=name + "_log_r2_local", value=log_r2_local)
        setattr(obj, name=name + "_r1_local_plus", value=r1_local_plus)
        setattr(obj, name=name + "_log_r2_local_plus", value=log_r2_local_plus)
        setattr(obj, name=name + "_r1_global", value=r1_global)
        setattr(obj, name=name + "_log_r2_global", value=log_r2_global)
        obj.list_w.append((name + "_w", w))
        obj.list_r1_local.append((name + "_r1_local", r1_local))
        obj.list_log_r2_local.append((name + "_log_r2_local", log_r2_local))
        obj.list_r1_local_plus.append((name + "_r1_local_plus", r1_local_plus))
        obj.list_log_r2_local_plus.append((name + "_log_r2_local_plus", log_r2_local_plus))
        obj.list_r1_global.append((name + "_r1_local", r1_global))
        obj.list_log_r2_global.append((name + "_log_r2_local", log_r2_global))

def hs_plus_raw(list_w,list_log_lam,list_log_tau,list_log_eta,nu):
    assert len(list_w) == len(list_log_lam) == len(list_log_tau)
    out = 0
    for i in range(len(list_w)):
        w = list_w[i]
        lam = torch.exp(list_log_lam[i])
        tau = torch.exp(list_log_tau[i])
        eta = torch.exp(list_log_eta[i])
        outw = -(w * w / (lam * lam * tau * tau * eta * eta)).sum() * 0.5
        outlam = -((nu + 1.) * 0.5 + torch.log(1 + (1 / nu) * (lam * lam))).sum()
        outeta = -((nu + 1.) * 0.5 + torch.log(1 + (1 / nu) * (eta * eta))).sum()
        outtau = -((1 + 1.) * 0.5 + torch.log(1 + (1 / 1) * (lam * lam))).sum()
        outhessian = - list_log_lam[i].sum()  - list_log_eta[i].sum() - list_log_tau[i].sum()
        out += outw + outlam + outeta + outtau + outhessian
    return(out)

def hs_plus_raw_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_w", value=[])
    setattr(obj, name="list_log_lam", value=[])
    setattr(obj, name="list_log_eta", value=[])
    setattr(obj, name="list_log_tau", value=[])
    for name, this_shape in obj.list_target_param_shape_dict:
        w = nn.Parameter(torch.zeros(this_shape))
        log_lam = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_eta = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_w", value=w)
        setattr(obj, name=name + "_log_lam", value=log_lam)
        setattr(obj, name=name + "_log_eta", value=log_eta)
        setattr(obj, name=name + "_log_tau", value=log_tau)
        obj.list_w.append((name + "_w", w))
        obj.list_log_lam.append((name + "_log_lam", log_lam))
        obj.list_log_lam.append((name + "_log_eta", log_lam))
        obj.list_log_tau.append((name + "_log_tau", log_tau))



def hs_plus_raw_ncp_prepare(obj):
    assert hasattr(obj, "list_target_param_shape_dict")
    setattr(obj, name="list_z", value=[])
    setattr(obj, name="list_log_lam", value=[])
    setattr(obj, name="list_log_eta", value=[])
    setattr(obj, name="list_log_tau", value=[])
    for name, this_shape in obj.list_target_param_shape_dict:
        z = nn.Parameter(torch.zeros(this_shape))
        log_lam = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_eta = nn.Parameter(torch.zeros(this_shape), requires_grad=True)
        log_tau = nn.Parameter(torch.zeros(1), requires_grad=True)
        setattr(obj, name=name + "_z", value=z)
        setattr(obj, name=name + "_log_lam", value=log_lam)
        setattr(obj, name=name + "_log_eta", value=log_eta)
        setattr(obj, name=name + "_log_tau", value=log_tau)
        obj.list_z.append((name + "_z", z))
        obj.list_log_lam.append((name + "_log_lam", log_lam))
        obj.list_log_lam.append((name + "_log_eta", log_lam))
        obj.list_log_tau.append((name + "_log_tau", log_tau))


def hs_plus_raw_ncp(list_z,list_log_lam,list_log_tau,list_log_eta,nu):
    assert len(list_z) == len(list_log_lam) == len(list_log_tau)
    out = 0
    for i in range(len(list_z)):
        z = list_z[i]
        lam = torch.exp(list_log_lam[i])
        tau = torch.exp(list_log_tau[i])
        eta = torch.exp(list_log_eta[i])
        outz = -(z*z).sum() * 0.5
        #outw = -(w * w / (lam * lam * tau * tau * eta * eta)).sum() * 0.5
        outlam = -((nu + 1.) * 0.5 + torch.log(1 + (1 / nu) * (lam * lam))).sum()
        outeta = -((nu + 1.) * 0.5 + torch.log(1 + (1 / nu) * (eta * eta))).sum()
        outtau = -((1 + 1.) * 0.5 + torch.log(1 + (1 / 1) * (lam * lam))).sum()
        outhessian = - list_log_lam[i].sum()  - list_log_eta[i].sum() - list_log_tau[i].sum()
        out += outz + outlam + outeta + outtau + outhessian
    return(out)

