import time
from abstract.abstract_class_point import point
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_nuts_util import *
#from abstract.abstract_static_sampler import *
from abstract.abstract_static_sampler import *
from abstract.metric import metric
from adapt_util.adapt_util import tuneable_param

from general_util.pytorch_util import welford_float


# this object is assumed to be initiated properly
class sampler_one_step(object):
    #def __init__(self,tune_dict,tune_param_obj_dict,init_point):
    def __init__(self,tune_dict,tune_param_objs_dict,init_point):
        #print(input_obj.input_dict)
        self.tune_param_objs_dict = tune_param_objs_dict
        # for param_name, obj in tune_param_objs_dict.items():
        #     val = obj.get_val()
        #     setattr(self,param_name,val)
        self.other_params = {}
        self.v_fun = tune_dict["v_fun"]
        self.dynamic = tune_dict["dynamic"]
        if self.dynamic:
            self.windowed = None
            if "max_tree_depth" in tune_dict:
                assert tune_dict["max_tree_depth"] > 0
                self.max_tree_depth = tune_dict["max_tree_depth"]
            else:
                self.max_tree_depth = 10

            self.other_params.update({"max_tree_depth":self.max_tree_depth})
        else:
            if "max_L" in tune_dict:
                self.max_L = tune_dict["max_L"]
            else:
                self.max_L = 1024
            self.other_params.update({"max_L":self.max_L})
            self.windowed = tune_dict["windowed"]
            assert self.windowed==True or self.windowed==False
        self.second_order = tune_dict["second_order"]
        self.metric_name = tune_dict["metric_name"]
        self.criterion = tune_dict["criterion"]
        precision_type = init_point.flattened_tensor.type()
        self.v_obj = self.v_fun(precision_type=precision_type)
        #self.v_obj.q_point = init_point
        self.v_obj.load_point(init_point)
        #if hasattr(tune_param_objs_dict,"alpha"):
        if "alpha" in tune_param_objs_dict:
            alpha_val = tune_param_objs_dict["alpha"].get_val()
            self.metric = metric(self.metric_name,self.v_obj,alpha_val)
        else:
            self.metric = metric(self.metric_name,self.v_obj)
        self.Ham = Hamiltonian(self.v_obj,self.metric)
        #if not self.dynamic:

        #if hasattr(tune_param_objs_dict,"evolve_t"):
        if "evolve_t" in tune_param_objs_dict:
            self.input_time=True
        #elif hasattr(tune_param_objs_dict,"evolve_L"):
        elif "evolve_L" in tune_param_objs_dict:
            self.input_time=False
        else:
            self.input_time=None
        self.ave_second_per_leapfrog = 0
        #self.one_step_function,self.tuneable_param = self.generate_sampler_one_step(self.windowed,self.dynamic,self.second_order,self.metric_name)
        # here self.one_step_function is raw_sampler_one_step
        self.one_step_function,self.tuneable_param = self.generate_sampler_one_step()
        # self.tuneable_param supplies the names of tuneable parameters that self.one_step_function needs
        #self.tune_param_obj_dict = tune_param_obj_dict
        #self.tuneable_param_obj_dict = {}
        #for name in self.tuneable_param:
        #    self.tuneable_param_obj_dict.update({name:tune_param_obj_dict})
        #for i in range(len(self.tuneable_param)):
        #    self.tuneable_param_dict.update({self.tuneable_param[i]:getattr(self,self.tuneable_param[i])})

        self.one_step_function = wrap(raw_sampler_one_step=self.one_step_function,other_parameters=self.other_params)
    def evolve(self):
        start = time.time()
        self.run()
        total_seconds = time.time() - start
        ave_seconds = total_seconds/self.log_obj["num_transitions"]
        self.ave_second_per_leapfrog,_ = self.welford_obj.mean_var(self.ave_second_per_leapfrog,ave_seconds)

    def set_tunable_param(self,metric):
        self.tuneable_param = tuneable_param(self.dynamic,self.second_order,metric,self.criterion,self.input_time)

    def find_ave_second_per_leapfrog(self):
        if self.ave_second_per_leapfrog==0:
            self.welford_obj = welford_float()
            for i in range(20):
                self.evolve()
        return(self.ave_second_per_leapfrog)

    def run(self):

        if hasattr(self,"log_obj"):
            #print("yes")
            out = self.one_step_function(input_point_obj=point(V=self.Ham.V),Ham_obj = self.Ham,
                                         tune_param_objs_dict=self.tune_param_objs_dict,log_obj=self.log_obj)
            #out = self.one_step_function(self.Ham.V.q_point,self.Ham,self.tuneable_param_dict,self.log_obj)
        else:
            out = self.one_step_function(input_point_obj=point(V=self.Ham.V), Ham_obj=self.Ham,
                                         tune_param_objs_dict=self.tune_param_objs_dict)
            #out = self.one_step_function(self.Ham.V.q_point, self.Ham, self.tuneable_param_dict)
        self.Ham.V.load_point(out[0])
        return(out[0].point_clone())



    def generate_sampler_one_step(self):
        if self.dynamic:
            if self.criterion == "nuts":
                out = abstract_NUTS
            elif self.criterion == "gnuts":
                out = abstract_GNUTS

            elif self.criterion == "xhmc":
                out = abstract_NUTS_xhmc
            else:
                raise ValueError("unknown criterion")
        else:
            if self.windowed:
                out = abstract_static_windowed_one_step
            else:
                out = abstract_static_one_step


        tuneable_par = tuneable_param(self.dynamic, self.second_order, self.metric_name, self.criterion, self.input_time)


        return (out,tuneable_par)


def wrap(raw_sampler_one_step,other_parameters):
    # want output to be function so that takes input point object and tune_param_dict
    # tune_par_setting = tuple = (Tunable,value,tune_by)
    # Tunable is a boolean variable . True means the variable will be tuned. False if fixed
    # value is the fixed param val if Tunable == False, initial value if Tunable == True
    # tune_by_dual is a boolean variable . True if we are tuning hte variable by dual averaging
    # false if by bayesian optimization
    # epsilon
    #if "epsilon" in fixed_tune_dict:
        #ep_setting = (False,fixed_tune_dict["epsilon"][0],fixed_tune_dict["epsilon"][1])
    #if "epsilon" in fixed_tune_dict:
    #    ep_setting = fixed_tune_dict["epsilon"]
    #else:
     #   ep_setting = tune_dict["epsilon"]
    def sampler_one_step(input_point_obj,Ham_obj,tune_param_objs_dict,log_obj=None):
        # tune_param_objs_dict contains tuning parameter objects for this integrator
        tune_param_dict = {}
        sampler_permissible_tune_parm = ("epsilon","evolve_t","evolve_L","xhmc_delta")
        for param_name,obj in tune_param_objs_dict.items():
            if param_name in sampler_permissible_tune_parm:
                tune_param_dict.update({param_name:obj.get_val()})
        if not log_obj is None:
            tune_param_dict.update({"log_obj":log_obj})
        tune_param_dict.update({"init_q":input_point_obj})
        tune_param_dict.update({"Ham":Ham_obj})
        if "max_tree_depth" in other_parameters:
            tune_param_dict.update({"max_tree_depth":other_parameters["max_tree_depth"]})

        if "max_L" in other_parameters:
            tune_param_dict.update({"max_L": other_parameters["max_L"]})
        #print(tune_param_dict)
        #exit()
        #print(tune_param_dict)
        #exit()
        return(raw_sampler_one_step(**tune_param_dict))

    return(sampler_one_step)



#def generate_sampler_one_step(Ham,windowed,dynamic,second_order,is_float,fixed_tune_dict,tune_dict):

 #   if is_float==True:
        #precision_type = 'torch.FloatTensor'
  #  else:
        #precision_type = 'torch.DoubleTensor'
        #
   # torch.set_default_tensor_type(precision_type)
    #input_time = False



    #out = wrap(windowed,fixed_tune_dict,tune_dict)
