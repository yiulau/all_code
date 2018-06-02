from abstract.T_dense_e import T_dense_e
from abstract.T_unit_e import T_unit_e
from abstract.T_diag_e import T_diag_e
from abstract.abstract_leapfrog_util import abstract_leapfrog_ult, windowerize
import torch
from general_util.time_diagnostics import time_diagnositcs


class Hamiltonian(object):
    # hamiltonian function
    def __init__(self,V,metric):
        self.V = V
        self.metric = metric
        if self.metric.name=="unit_e":
            T_obj = T_unit_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult
        elif self.metric.name=="diag_e":
            T_obj = T_diag_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult
        elif self.metric.name=="dense_e":
            T_obj = T_dense_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult


        self.windowed_integrator = windowerize(self.integrator)

        self.T = T_obj
        self.dG_dt = self.setup_dG_dt()
        self.p_sharp_fun = self.setup_p_sharp()
        self.diagnostics = time_diagnositcs()
        self.V.diagnostics = self.diagnostics


    def evaluate_all(self,q_point=None,p_point=None):
        # returns (H,V,T)
        self.V.load_point(q_point)
        self.T.load_point(p_point)
        out = [0,self.V.evaluate_scalar(),self.T.evaluate_scalar()]
        out[0] = out[1]+out[2]
        self.diagnostics.add_num_H_eval(1)
        return(out)
    def evaluate(self,q_point=None,p_point=None):
        #print(q_point.flattened_tensor)
        #print(p_point.flattened_tensor)
        self.V.load_point(q_point)
        self.T.load_point(p_point)
        #print(q_point.list_tensor)
        #print(self.V.flattened_tensor)
        #print(self.T.flattened_tensor)
        out = self.V.evaluate_scalar() + self.T.evaluate_scalar()
        #self.diagnostics.add_num_H_eval(1)
        return(out)

    def setup_dG_dt(self):

        if(self.metric.name=="dense_e" or self.metric.name=="diag_e" or self.metric.name=="unit_e"):
            def dG_dt(q,p):
                self.T.load_point(p)
                return (2 * self.T.evaluate_scalar() - torch.dot(q.flattened_tensor, self.V.dq(q.flattened_tensor)))
        else:
            raise ValueError("unknown metric name")
        return(dG_dt)
    def setup_p_sharp(self):
        if(self.metric.name=="dense_e" or self.metric.name=="unit_e" or self.metric.name=="diag_e"):
            def p_sharp(q,p):
                out = p.point_clone()
                out.flattened_tensor.copy_(self.T.dp(p.flattened_tensor))
                out.load_flatten()
                return(out)

        else:
            raise ValueError("unknown metric name")
        return(p_sharp)
