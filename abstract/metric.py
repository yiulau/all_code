import numpy,torch,math

class metric(object):
    # should store information like Cov, vec. Whether its for flattened tensor defined V. The cov and var should be in
    # flattened version if necessary
    # stores alpha for softabs
    def __init__(self,name,V_instance,alpha=None):

        self.name = name

        if self.name=="unit_e":
            pass
        elif self.name=="diag_e":
            self.num_var = V_instance.num_var
            self.store_shapes = V_instance.store_shapes
            self.store_lens = V_instance.store_lens
            self._sd_list_tensor = numpy.empty(self.num_var, dtype=type(V_instance.flattened_tensor))
            for i in range(self.num_var):
                self._sd_list_tensor[i] = torch.ones(self.store_shapes[i])
            self._var_list_tensor = numpy.empty(self.num_var,dtype=type(V_instance.list_var[0]))
            for i in range(self.num_var):
                 self._var_list_tensor[i] = torch.ones(self.store_shapes[i])
            self._flattened_var = torch.ones(V_instance.dim)
            self._flattened_sd = torch.ones(V_instance.dim)
        elif name=="dense_e":
            # covL * covL^T = cov
            self._flattened_cov_L = torch.eye(V_instance.dim,V_instance.dim)
            self._flattened_cov_inv = torch.eye(V_instance.dim,V_instance.dim)
        # elif name=="softabs" or name=="softabs_diag" or name=="softabs_outer_product" or name=="softabs_outer_product_diag":
        #     if alpha==None:
        #         raise ValueError("alpha needs be defined for softabs metric")
        #     elif alpha <= 0 or alpha==math.inf:
        #         raise ValueError("alpha needs be > 0 and less than < Inf")
        #     self.msoftabsalpha = alpha

        else:
            raise ValueError("unknown metric type")
    def set_metric(self,input_var):
        # input: either flattened empircial covariance for dense_e or
        # flattened var tensor for diag_e
        if self.name == "diag_e":
            try:
                # none of the variances or negative
                assert not sum(input_var < 0) > 0
                # none of the variances are too small or too large
                assert not sum(input_var < 1e-8) > 0 and not sum(input_var >1e8) > 0
                self._flattened_var.copy_(input_var)
                self._flattened_sd.copy_(torch.sqrt(self._flattened_var))
                self._load_flatten()
            except:
                raise ValueError("negative var or extreme var values")
        elif self.name == "dense_e":
            try:
                temp_cov_inv = torch.inverse(input_var)
                self._flattened_cov.copy_(input_var)
                self._flattened_cov_inv.copy_(temp_cov_inv)
            except:
                raise ValueError("not decomposable")

        else:
            raise ValueError("should not use this function unless the metrics are diag_e or dense_e")

    def _load_flatten(self):
        if not self.name =="diag_e":
            raise ValueError("should not use this fun if not diag_e")
        else:
            cur = 0
            for i in range(self.num_var):
                self._var_list_tensor[i].copy_(self._flattened_cov[cur:(cur + self.store_lens[i])].view(self.store_shapes[i]))
                self._sd_list_tensor[i].copy_(torch.sqrt(self._var_list_tensor[i]))
                cur = cur + self.store_lens[i]

