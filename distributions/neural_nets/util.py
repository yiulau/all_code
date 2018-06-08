import torch,numpy
import torch.nn as nn
from general_util.pytorch_random import log_inv_gamma_density

def prediction_error(target_dataset,v_nn_obj,type):
    # target_dataset["input"] and target_dataset["output"] need to be tensors
    permissible_types = ("classification","regression")
    assert isinstance(target_dataset, dict)
    assert "input" in target_dataset
    assert "target" in target_dataset
    assert target_dataset["input"].shape[0] == len(target_dataset["output"])
    assert type in permissible_types

    if type=="classification":
        out_prob = v_nn_obj.predict(target_dataset["input"])
        max_prob,predicted = torch.max(out_prob,1)
        error = (predicted !=target_dataset["output"].type("torch.LongTensor")).sum()/len(predicted)
    elif type == "regression":
        predicted = v_nn_obj.predict(target_dataset["input"])
        error = ((predicted - target_dataset["output"])*(predicted - target_dataset["output"])).sum()
    else:
        raise ValueError("unknown type")

    return(error)


def posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type):
    # output store_dist [input_sample i , class probability j , mcmc sample k ] for classification case
    # output store_dist [input_sample i , predicted y , mcmc sample k ]
    assert target_dataset["input"].shape[0] == len(target_dataset["output"])
    permissible_types = ("classification", "regression")
    assert type in permissible_types
    num_mcmc_samples = mcmc_samples.shape[0]
    num_target_samples = target_dataset["input"].shape[0]
    if type =="classifcation":
        test_samples = target_dataset["input"][0:1,:]
        out_prob = v_nn_obj.predict(test_samples)
        num_classes = out_prob.shape[1]
        # code above only to get number of classes
        store_dist = torch.zeros(num_target_samples,num_classes,num_mcmc_samples)
        for i in range(num_mcmc_samples):
            store_dist[:,:,i].copy_(v_nn_obj.predict(target_dataset["input"]))

    elif type =="regression":
        store_dist = torch.zeros(num_target_samples,num_mcmc_samples)
        for i  in range(num_mcmc_samples):
            store_dist[:,i].copy_(v_nn_obj.predict(target_dataset["input"]))

    return(store_dist)

def map_prediction(target_dataset,v_nn_obj,mcmc_samples,type,memory_efficient=False):
    # returns map if classification, posterior mean if classification
    # also returns mean,var, mcse, eff for all relevant quantities , class prob for classification, predicted y for
    # regression
    num_mcmc_samples = mcmc_samples.shape[0]
    num_target_samples = target_dataset["input"].shape[0]

    if type=="classification":
        if memory_efficient:
            store_prediction = torch.zeros(num_target_samples,2).type("torch.LongTensor")
            for i in range(num_target_samples):
                test_samples = target_dataset["input"][0:1, :]
                out_prob = v_nn_obj.predict(test_samples)
                num_classes = out_prob.shape[1]
                # first column stores mean, second column stores var
                store_prob = torch.zeros(num_classes,4)
                temp = torch.zeros(num_mcmc_samples,num_classes)
                for j in range(num_mcmc_samples):
                    new_sample = target_dataset["input"][i:i+1,:]
                    out_prob = v_nn_obj.predict(new_sample)
                    temp[i,:].copy_(out_prob)
                store_prob[:,0] =  temp.mean(dim=0)
                store_prob[:,1] = temp.var(dim=0)
                max_prob,map_class  = torch.max(store_prob[:,:1],0)
                store_prediction[i,0] = map_class[0]
                store_prediction[i,1] = store_prob[map_class[0],1]

        else:
            store_prediction = torch.zeros(num_mcmc_samples,2).type("torch.LongTensor")
            post_dist = posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type)
            post_class_prob_mean = post_dist.mean(dim=2)
            post_class_prob_var = post_dist.var(dim=2)
            map_prob,map_class  = torch.max(post_class_prob_mean,dim=1)
            store_prediction[:,0].copy_(map_class)
            store_prediction[:,1].copy_(post_class_prob_var[map_class,1])
    elif type=="regression":
        if memory_efficient:
            store_prediction = torch.zeros(num_target_samples,2)
            for i in range(num_target_samples):
                temp = []*num_mcmc_samples
                for j in range(num_mcmc_samples):
                    new_sample = target_dataset["input"][i:i+1,:]
                    out_predicted = v_nn_obj.predict(new_sample)
                    temp[j] = out_predicted

                store_prediction[i,0] = numpy.mean(temp)
                store_prediction[i,1] = numpy.var(temp)
        else:
            store_prediction = torch.zeros(num_target_samples,2)
            post_dist = posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type)
            post_out_mean  = torch.mean(post_dist,dim=1)
            post_out_var = torch.var(post_dist,dim=1)
            store_prediction.copy_(post_out_mean)

    return(store_prediction)



def gamma_density(x,alpha,beta):
     out =  (alpha-1)*torch.log(x) - x/beta
     return(out)


class setup_parameter(object):
    def __init__(self,obj,name,shape,prior_obj):
        self.prior_obj = prior_obj
        assert not hasattr(obj,name)
        if prior_obj.model_param=="cp":
            self.w_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
        else:
            self.z_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)

        if prior_obj.name == "hs":
            if prior_obj.prior_cp=="ncp":
                #self.z_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
                self.local_r1_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
                self.log_local_r2_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
                self.global_r1_obj = nn.Parameter(torch.zeros(1),requires_grad=True)
                self.log_global_r2_obj = nn.Parameter(torch.zeros(1),requires_grad=True)

            else:
                self.local_lamb_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
                self.global_lamb_obj = nn.Parameter(torch.zeros(1),requires_grad=True)

        elif prior_obj.name =="rhorseshoe":
            if prior_obj.prior_cp=="ncp":
                #self.z_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
                self.local_r1_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
                self.log_local_r2_obj = nn.Parameter(torch.zeros(shape), requires_grad=True)
                self.global_r1_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.log_global_r2_obj = nn.Parameter(torch.zeros(1), requires_grad=True)
                self.c_z = nn.Parameter(torch.zeros(1),requires_grad=True)
                self.c_log_tau = nn.Parameter(torch.zeros(1),requires_grad=True)
            else:
                self.local_lamb_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)
                self.global_lamb_obj = nn.Parameter(torch.zeros(shape),requires_grad=True)

        else:
            raise ValueError("unknown prior")

    def get_val(self):
        if self.prior_obj["name"]=="horseshoe_ncp":
            lamb = torch.sqrt(torch.exp(self.log_local_r2_obj))*self.local_r1_obj
            tau = torch.sqrt(torch.exp(self.log_global_r2_obj))*self.global_r1_obj
            w = self.z_obj * lamb*tau

        elif self.prior_obj["name"]=="rhorseshoe":
            c = self.c_z * torch.exp(self.c_log_tau)
            lamb = torch.sqrt(torch.exp(self.log_local_r2_obj)) * self.local_r1_obj
            tau = torch.sqrt(torch.exp(self.log_global_r2_obj)) * self.global_r1_obj
            lamb_tilde = c*c * lamb*lamb/(c*c + tau*tau*lamb*lamb)
            w = self.z_obj * lamb_tilde*tau

        return(w)

    def get_out(self):
        if self.prior_obj["name"]=="hs":
            if self.prior_obj.hyper_param=="ncp":
                z_out = (self.z_obj*self.z_obj).sum()
                local_r1_out = (self.local_r1_obj*self.local_r1_obj).sum()
                global_r1_out = (self.global_r1_obj*self.global_r1_obj).sum()
                local_r2_out = log_inv_gamma_density(torch.exp(self.log_local_r2_obj),0.5+1,0.5+1) + self.log_local_r2_obj
                local_r2_out = local_r2_out.sum()
                global_r2_out = log_inv_gamma_density(torch.exp(self.log_global_r2_obj),0.5+1,0.5+1) + self.log_global_r2_obj
                global_r2_out = global_r2_out.sum()
                out = z_out + local_r1_out + global_r1_out + local_r2_out + global_r2_out

        elif self.prior_obj["name"] == "rhorseshoe":
            if self.prior_obj.hyper_param=="ncp":
                c_z_out = (self.c_z * self.c_z).sum() * 0.5
                c_tau = torch.exp(self.c_log_tau)
                c_tau_out = log_inv_gamma_density(c_tau, 1, 1) + self.c_log_tau
                z_out = (self.z_obj * self.z_obj).sum()
                local_r1_out = (self.local_r1_obj * self.local_r1_obj).sum()
                global_r1_out = (self.global_r1_obj * self.global_r1_obj).sum()
                local_r2_out = log_inv_gamma_density(torch.exp(self.log_local_r2_obj), 0.5 + 1,
                                                     0.5 + 1) + self.log_local_r2_obj
                local_r2_out = local_r2_out.sum()
                global_r2_out = log_inv_gamma_density(torch.exp(self.log_global_r2_obj), 0.5 + 1,
                                                      0.5 + 1) + self.log_global_r2_obj
                global_r2_out = global_r2_out.sum()
                out = z_out + local_r1_out + global_r1_out + local_r2_out + global_r2_out

        return(out)


class prior(object):

    def __init__(self,name,model_param,hyper_param):
        self.permissible_values = (
        "hs", "rhorseshoe", "horseshoe_ard", "rhorseshoe_ard", "gaussian_inv_gamma_ard""gaussian_inv_gamma",
        "normal")
        assert name in self.permissible_values
        self.name=name
        self.model_param = model_param
        self.hyper_param = hyper_param


