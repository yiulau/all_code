from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
import torch

class T_unit_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_unit_e, self).__init__(linkedV)
        #return()

    def evaluate_scalar(self,q_point=None,p_point=None):
        if not q_point is None:
            raise ValueError("should not pass q_point for this metric")
            pass
        if not p_point is None:
            self.load_point(p_point)
        output = 0
        for i in range(len(self.list_var)):
            output += (self.list_var[i].data * self.list_var[i].data).sum() * 0.5
        return(output)

    def dp(self,p_flattened_tensor):
        out = p_flattened_tensor
        return(out)
    def dtaudp(self):
        return (self.p)

    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self,q):
        out = point(list_tensor=self.list_tensor,pointtype="p",need_flatten=self.need_flatten)
        out.flattened_tensor.copy_(torch.randn(self.dim))
        out.load_flatten()
        return(out)
