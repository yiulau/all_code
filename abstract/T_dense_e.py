from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
import torch
class T_dense_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_dense_e, self).__init__(linkedV)

    def evaluate_scalar(self,q_point=None,p_point=None):
        if not q_point is None:
            print("should not pass q_point for this metric")
            pass
        if not p_point is None:
            self.load_point(p_point)
        temp = torch.mv(self.metric._flattened_cov_inv, self.flattened_tensor)
        output = torch.dot(temp, self.flattened_tensor) * 0.5
        return(output)



    # def dtaudp(self,p=None):
    #     if self.need_flatten:
    #         self.load_listp_to_flattened(self.p, self.flattened_p_tensor)
    #         temp = torch.mv(self.metric._flattened_cov, self.flattened_p_tensor)
    #         self.load_flattened_tenosr_to_target_list(self.gradient,self.flattened_p_tensor)
    #         return (self.gradient)
    #     else:
    #         self.load_flattened_tenosr_to_target_list(self.gradient,torch.mv(self.metric._flattened_cov, self.p[0]))
    #         return (self.gradient)

    def dp(self,p_flattened_tensor):
        #print(self.metric._flattened_cov)
        out = torch.mv(self.metric._flattened_cov_inv, p_flattened_tensor)
        return(out)
    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self,q):
        out = point(list_tensor=self.list_tensor,pointtype="p",need_flatten=self.need_flatten)
        #print(self.metric._flattened_covL)
        out.flattened_tensor.copy_(torch.mv(self.metric._flattened_cov_L, torch.randn(self.dim)))
        out.load_flatten()
        return(out)