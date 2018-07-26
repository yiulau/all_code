import torch,numpy,os,math
import torch.nn as nn
from distributions.bayes_model_class import bayes_model_class
from torch.autograd import Variable
from distributions.neural_nets.priors.prior_util import prior_generator

# hierarchical prior for input to hidden units, scale = sqrt(1/input_dim)
#  normal prior for hidden to output units with variance 1/num_hidden_units

class V_fc_model_1(bayes_model_class):
    def __init__(self,input_data,precision_type,prior_dict,model_dict):
        self.prior_dict = prior_dict
        self.model_dict = model_dict
        super(V_fc_model_1, self).__init__(input_data=input_data,precision_type=precision_type)
    def V_setup(self):
        self.dim = self.input_data["input"].shape[1]
        self.num_ob = self.input_data["target"].shape[0]
        self.num_classes = len(numpy.unique(self.input_data["target"]))
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.num_units = self.model_dict["num_units"]
        prior_hidden_fn = prior_generator(self.prior_dict["name"])
        prior_out_fn = prior_generator("normal")
        self.hidden_in = prior_hidden_fn(obj=self,name="hidden_in",shape=(self.num_units,self.dim),global_scale=1)
        self.hidden_out = prior_out_fn(obj=self,name="hidden_out",shape=(self.num_classes,self.num_units),global_scale=1)




        self.y = Variable(torch.from_numpy(self.input_data["target"]),requires_grad=False).type("torch.LongTensor")
        self.X = Variable(torch.from_numpy(self.input_data["input"]),requires_grad=False).type(self.precision_type)
        self.dict_parameters = {"hidden_in": self.hidden_in,"hidden_out":self.hidden_out}
        # include

        return()

    def forward(self,input=None):
        if input is None:
            X = self.X
            y = self.y
        else:
            X = Variable(input["input"],requires_grad=False).type(self.precision_type)
            y = Variable(input["target"],requires_grad=False).type("torch.LongTensor")

        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()

        criterion = nn.CrossEntropyLoss()

        neg_log_likelihood = criterion(out_units,y)

        hidden_in_out = self.hidden_in.get_out()
        hidden_out_out = self.hidden_out.get_out()

        prior = hidden_in_out + hidden_out_out #+ in_sigma_out + out_sigma_out

        neg_logposterior = -prior  + neg_log_likelihood
        out = neg_logposterior

        return(out)

    def predict(self,inputX):
        X = Variable(torch.from_numpy(inputX),requires_grad=False).type(self.precision_type)
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        softmax = nn.Softmax(dim=-1)
        prob = softmax(out_units)
        return(prob.data)

    def log_p_y_given_theta(self, observed_point, posterior_point):
        self.load_point(posterior_point)
        X = observed_point["input"]
        y = observed_point["target"]
        hidden_units = torch.tanh((self.hidden_in.get_val().mm(X.t())))
        out_units = self.hidden_out.get_val().mm(hidden_units).t()
        criterion = nn.CrossEntropyLoss()
        neg_log_likelihood = criterion(out_units, self.y)
        out = -neg_log_likelihood
        return(out)


import torch, numpy,math
def sghmc_one_step(init_q_point,epsilon,L,Ham,alpha,eta,betahat,input_data,adjust_factor):
    # eta is the learning rate
    # adjust_factor should be full_data_size/batch_size
    #print(adjust_factor)
    #exit()
    init_v_point = init_q_point.point_clone()

    v = init_v_point.point_clone()
    q = init_q_point.point_clone()
    dim = len(init_q_point.flattened_tensor)
    v.flattened_tensor.copy_(torch.randn(dim)*epsilon)
    explode_grad = False
    try:
        for i in range(L):
            q.flattened_tensor += v.flattened_tensor
            q.load_flatten()
            noise = torch.randn(dim)
            grad,explode_grad = Ham.V.dq(q_flattened_tensor=q.flattened_tensor,input_data=input_data)
            v_val = Ham.V.forward()
            #print("v {}".format(v_val))
            if not explode_grad:
                grad = grad*adjust_factor
                delta_v = -eta*grad - alpha * v.flattened_tensor + math.sqrt(2*(alpha-betahat))*noise
                v.flattened_tensor += delta_v
                v.load_flatten()
            else:
                break
    except:
        q = None
        explode_grad = True

    return(q,explode_grad)


def sghmc_sampler(init_q_point,epsilon,L,Ham,alpha,eta,betahat,full_data,num_samples,thin,burn_in,batch_size):
    dim = len(init_q_point.flattened_tensor)
    full_data_size = len(full_data["target"])
    full_data = {"input":torch.from_numpy(full_data["input"]),"target":torch.from_numpy(full_data["target"])}
    if thin>0:
        store= torch.zeros(round(num_samples/thin),dim)
    else:
        assert thin==0
        store = torch.zeros(num_samples,dim)
    cur = thin
    store_i = 0
    explode_grad = False
    q = init_q_point.point_clone()
    for i in range(num_samples):
        input_data = subset_data(full_data=full_data,batch_size=batch_size)
        q,explode_grad = sghmc_one_step(q,epsilon,L,Ham,alpha,eta,betahat,input_data,full_data_size/batch_size)
        #print(q.flattened_tensor)
        if not explode_grad:
            print(q.flattened_tensor)
            if i >= burn_in:
                cur -= 1
                if not cur > 0.1:
                    keep = True
                    store_i += 1
                    cur = thin
                else:
                    keep = False
                store[store_i-1,:].copy_(q.flattened_tensor.clone())
        else:
            break

    return(store,explode_grad)




def subset_data(full_data,batch_size):
    full_data_size = len(full_data["target"])
    chosen_indices = list(numpy.random.choice(a=full_data_size, size=batch_size, replace=False))
    chosen_indices = [numpy.asscalar(v) for v in chosen_indices]
    subset_input = full_data["input"][chosen_indices,:]
    subset_target = full_data["target"][chosen_indices]
    out = {"input":subset_input,"target":subset_target}
    return(out)


