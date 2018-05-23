from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
import torch
class T_diag_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_diag_e, self).__init__(linkedV)

    def evaluate_scalar(self,q_point=None,p_point=None):
        if not q_point is None:
            print("should not pass q_point for this metric")
            pass
        if not p_point is None:
            self.load_point(p_point)
        output = 0
        for i in range(len(self.list_var)):
            output += (self.list_var[i].data * self.list_var[i].data*self.metric._var_list_tensor[i]).sum() * 0.5

        return(output)
    def dp(self,p_flattened_tensor):
        out = self.metric._var_vec * p_flattened_tensor
        return(out)
    def dtaudp(self,p=None):
        if p==None:
            for i in range(len(self.list_shapes)):
                self.gradient[i].copy_(self.metric_var_list[i] * self.p[i])
        else:
            for i in range(len(self.list_shapes)):
                self.gradient[i].copy_(self.metric_var_list[i] * p[i])
        return (self.gradient)

    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self,q):
        out = point(list_tensor=self.list_tensor,pointtype="p",need_flatten=self.need_flatten)
        out.flattened_tensor.copy_(self.metric._sd_vec * torch.randn(self.dim))
        out.load_flatten()
        return(out)