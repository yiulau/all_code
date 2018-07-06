import abc, numpy, pickle, os,copy
from abstract.mcmc_sampler import mcmc_sampler_settings_dict,mcmc_sampler
from adapt_util.tune_param_classes.tune_param_setting_util import *

def experiment_setting_dict(chain_length,
                            tune_l,warm_up,num_chains_per_sampler=1,num_cpu_per_sampler=1,
                            is_float=False,thin=1,allow_restart=False,max_num_restarts=5):
    out = {"chain_length":chain_length}
    out.update({"num_chains_per_sampler":num_chains_per_sampler})
    out.update({"allow_restart":allow_restart,"max_num_restarts":max_num_restarts,"is_float":is_float,"thin":thin})
    out.update({"num_cpu_per_sampler":num_cpu_per_sampler,"tune_l":tune_l,"warm_up":warm_up})


    return(out)
# def resume_experiment(save_name):
#     experiment_obj = pickle.load(open(save_name, 'rb'))
#     experiment_obj.run()
#     return()
def get_tune_settings_dict(tune_dict):
    v_fun = tune_dict["v_fun"]
    ep_dual_metadata_argument = {"name": "epsilon", "target": 0.8, "gamma": 0.05, "t_0": 10,
                                 "kappa": 0.75, "obj_fun": "accept_rate", "par_type": "fast"}
    adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow", dim=v_fun(
        precision_type="torch.DoubleTensor").get_model_dim())]

    if "cov" in tune_dict:
        if "adapt"==tune_dict["cov"]:
            adapt_cov_arguments = adapt_cov_arguments
        else:
            adapt_cov_arguments = []
    else:
        adapt_cov_arguments = []

    if tune_dict["epsilon"]=="dual":
        dual_args_list = [ep_dual_metadata_argument]
    else:
        dual_args_list = []
    other_arguments = other_default_arguments()

    tune_settings_dict = tuning_settings(dual_arguments=dual_args_list,opt_arguments=[],adapt_cov_arguments=adapt_cov_arguments,
                                         other_arguments=other_arguments)
    return(tune_settings_dict)

class experiment(object):
    #__metaclass__ = abc.ABCMeta
    def __init__(self,input_object=None,experiment_setting=None,fun_per_sampler=None):
        # fun per sampler process sampler to extract desired quantities
        self.fun_per_sampler = fun_per_sampler
        self.experiment_setting = experiment_setting
        self.input_object = input_object
        self.tune_param_grid = self.input_object.tune_param_grid
        self.store_grid_obj = numpy.empty(self.input_object.grid_shape,dtype=object)
        self.experiment_result_grid_obj = numpy.empty(self.input_object.grid_shape,dtype=object)
        self.input_name_list = self.input_object.param_name_list
        #loop through each point in the grid and initiate an sampling_object
        it = numpy.nditer(self.store_grid_obj, flags=['multi_index',"refs_ok"])
        cur = 0
        self.id_to_multi_index = []
        self.multi_index_to_id = {}
        while not it.finished:

            self.id_to_multi_index.append(it.multi_index)
            self.multi_index_to_id.update({it.multi_index: cur})
            tune_dict = self.tune_param_grid[it.multi_index]
            sampling_metaobj = mcmc_sampler_settings_dict(mcmc_id = cur,
                                                          samples_per_chain= self.experiment_setting["chain_length"],
                                                          num_chains=self.experiment_setting["num_chains_per_sampler"],
                                                          num_cpu=self.experiment_setting["num_cpu_per_sampler"],
                                                          thin=self.experiment_setting["thin"],
                                                          tune_l_per_chain=self.experiment_setting["tune_l"],
                                                          warmup_per_chain=self.experiment_setting["warm_up"],
                                                          is_float=self.experiment_setting["is_float"],
                                                          allow_restart=self.experiment_setting["allow_restart"],
                                                          max_num_restarts=self.experiment_setting["max_num_restarts"])
            tune_settings_dict = get_tune_settings_dict(tune_dict)
            grid_pt_metadict = {"mcmc_id":cur,"started":False,"completed":False,"saved":False}
            self.store_grid_obj[it.multi_index] = {"sampler":mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=sampling_metaobj,tune_settings_dict=tune_settings_dict),"metadata":grid_pt_metadict}
            self.experiment_result_grid_obj[it.multi_index] = {}
            it.iternext()
            cur +=1



    def pre_experiment_diagnostics(self,test_run_chain_length=15):

    # 1 estimated total volume
    # 2 estimated  computing time (per chain )
    # 3 estimated total computing time (serial)
    # 4 estimated total coputing time (if parallel, given number of agents)
    # 5 ave number of time per leapfrog

        experiment_setting = experiment_setting_dict(chain_length=test_run_chain_length)
        temp_experiment = self.experiment.clone()

        temp_output = temp_experiment.run()
        out = {}
        time = temp_output.sampling_metadata.total_time
        ave_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
        total_size = temp_output.sample_metadata.size_mb
        estimated_total_volume = total_size * (self.num_chains * self.num_per_chain) / test_run_chain_length
        out.append({"total_volume": estimated_total_volume})
        estimated_compute_seconds_per_chain = time * self.num_per_chain / test_run_chain_length
        out.append({"seconds_per_chain": estimated_compute_seconds_per_chain})
        estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
        out.append({"total_seconds": estimated_compute_seconds})
        estimated_compute_seconds_parallel = estimated_compute_seconds / self.num_agents
        out.append({"total_seconds with parallel": estimated_compute_seconds_parallel})
        with open('model.pkl', 'wb') as f:
            pickle.dump(temp_output, f)
        size = os.path.getsize("./model.pkl") / (1024. * 1024)
        os.remove("./model.pkl")


    def run(self):
        it = numpy.nditer(self.store_grid_obj, flags=['multi_index', "refs_ok"])
        while not it.finished:
            #self.store_grid_obj[it.multi_index]["metadata"]
            sampler = self.store_grid_obj[it.multi_index]["sampler"]
            self.store_grid_obj[it.multi_index]["metadata"].update({"started": True})
            sampler.start_sampling()
            fun_output,output_names = self.fun_per_sampler(sampler)
            self.experiment_result_grid_obj[it.multi_index].update({"fun_output": fun_output})
            self.experiment_result_grid_obj[it.multi_index].update({"output_names": output_names})
            self.store_grid_obj[it.multi_index]["metadata"].update({"completed": True,"saved":True})
            #self.saves_progress()
            it.iternext()

        # save_name = self.experiment_setting["save_name"]
        # with open(save_name, 'wb') as f:
        #     pickle.dump(self.experiment_result_grid_obj, f)
        return()

    def run_specific(self,list_of_multi_index_id=None,list_mcmc_id=None):
        assert not list_of_multi_index_id is None or not list_mcmc_id is None
        if list_of_multi_index_id is None:
            is_mcmc_id = True
            id_list = list_mcmc_id
        else:
            is_mcmc_id = False
            id_list = list_of_multi_index_id
        for id in id_list:
            if is_mcmc_id:
                input_id = self.id_to_multi_index[id]
            else:
                input_id = id
            sampler = self.store_grid_obj[input_id]["sampler"]
            self.store_grid_obj[input_id]["metadata"].update({"started":True})
            sampler.start_sampling()
            fun_output = self.fun_per_sampler(ran_sampler=sampler)
            self.experiment_result_grid_obj[input_id].update({"fun_output": fun_output})
            self.store_grid_obj[input_id]["metadata"].update({"completed": True,"saved":True})
            #self.saves_progress()

        # save_name = self.experiment_setting["save_name"]
        # with open(save_name, 'wb') as f:
        #     pickle.dump(self.experiment_result_grid_obj, f)
        return()
    def clone(self):
        # clone experiment object at pre-sampling state
        out = experiment(input_object=self.input_object.clone(),experiment_setting=copy.deepcopy(self.experiment_setting))
        return(out)
    # def saves_progress(self):
    #     save_name = self.experiment_setting["save_name"]
    #     with open(save_name, 'wb') as f:
    #         pickle.dump(self, f)
    def np_diagnostics(self):
        it = numpy.nditer(self.store_grid_obj, flags=['multi_index', "refs_ok"])
        np_diagnostics,diagnostics_name = self.store_grid_obj[it.multi_index]["sampler"].np_diagnostics()
        store_shape = self.store_grid_obj.shape + np_diagnostics.shape
        np_store = numpy.zeros(store_shape)
        while not it.finished:
            # self.store_grid_obj[it.multi_index]["metadata"]
            output,_ = self.store_grid_obj[it.multi_index]["sampler"].np_diagnostics()
            new_index = list(it.multi_index) + [...]
            np_store[new_index] = output
            # self.saves_progress()
            it.iternext()
        return(np_store,diagnostics_name)
    def np_output(self):
        col_names = self.input_name_list
        it = numpy.nditer(self.experiment_result_grid_obj, flags=['multi_index', "refs_ok"])
        output_dim = len(self.experiment_result_grid_obj[it.multi_index]["fun_output"])
        output_names = self.experiment_result_grid_obj[it.multi_index]["output_names"]
        col_names += ["output"]
        store_shape = list(self.experiment_result_grid_obj.shape) + [output_dim]
        np_store = numpy.zeros(store_shape)
        while not it.finished:
            # self.store_grid_obj[it.multi_index]["metadata"]
            output = self.experiment_result_grid_obj[it.multi_index]["fun_output"]
            new_index = list(it.multi_index) + [...]
            np_store[new_index] = output

            # self.saves_progress()
            it.iternext()
        return(np_store,col_names,output_names)




#input_dict = {"v_fun":[1,2],"alpha":[0],"epsilon":[0.1,0.2,0.3],"second_order":[True,False]}

#input_obj = tuneinput_class()

#exper_obj = experiment(input_object=input_object)

#print(exper_obj.grid_shape)
#print(exper_obj.param_name_list)
#exper_obj2 = exper_obj.clone()
#exit()
#print(exper_obj.ep)

#print(exper_obj.__dict__)
#exit()
#print(experiment.__dict__)



# class experiment_meta(object):
#     def __init__(self,chain_length):
#         self.chain_length = chain_length
#         self.warmup_per_chain



#class experiment_meta(object):
    #def __init__(self):


class tuneinput_class(object):
    permissible_var_names = ("v_fun", "dynamic", "windowed", "second_order", "criterion", "metric_name", "epsilon", "evolve_t",
                             "evolve_L", "alpha", "xhmc_delta", "cov","max_tree_depth")

    permissible_var_values = {"dynamic": (True, False)}
    permissible_var_values.update({"windowed":(True,False)})
    permissible_var_values.update({"second_order": (True, False)})
    permissible_var_values.update({"criterion": ("nuts", "gnuts", "xhmc",None)})
    permissible_var_values.update({"metric_name": ("unit_e", "diag_e", "dense_e", "softabs_e",
                                              "softabs_diag", "softabs_op", "softabs_op_diag")})
    permissible_var_values.update({"epsilon": ("dual", "opt")})
    permissible_var_values.update({"evolve_t": ("dual", "opt", None)})
    permissible_var_values.update({"evolve_L": ("dual", "opt", None)})
    permissible_var_values.update({"alpha": ("dual", "opt", None)})
    permissible_var_values.update({"xhmc_delta": ("dual", "opt", None)})
    permissible_var_values.update({"cov": ("adapt", None)})

    def __init__(self,input_dict=None):
        self.input_dict = input_dict
        self.grid_shape = []
        self.param_name_list = []
        for param_name,val_list in input_dict.items():
            if not param_name in self.permissible_var_names:
                #print(param_name)
                raise ValueError("not one of permissible attributes")
            elif len(val_list)==0:
                raise ValueError("can't have empty list ")
            else:
                if param_name=="v_fun":
                    pass
                else:
                    for option in val_list:
                        pass
                        #if not "fixed" in self.permissible_var_values[param_name]:
                            #if not option in self.permissible_var_values[param_name]:
                              #  print(param_name)
                              #  print(option)
                              #  raise ValueError("not one of permitted options for this attribute")
            setattr(self,param_name,val_list)
            #self.grid_shape.append(len(val_list))
            #self.param_name_list.append(param_name)

        for name in self.permissible_var_names:
            if hasattr(self,name):
                self.grid_shape.append(len(getattr(self,name)))
                self.param_name_list.append(name)

        self.tune_param_grid = numpy.empty(self.grid_shape)



        # store tuning parameters value in a grid
        self.tune_param_grid = numpy.empty(self.grid_shape,dtype=object)
        #print(self.tune_param_grid[0,0,0,0])
        it = numpy.nditer(self.tune_param_grid,flags=["multi_index","refs_ok"])
        while not it.finished:
            settings_dict = {}
            for i in range(len(self.param_name_list)):
                settings_dict.update({self.param_name_list[i]:getattr(self,self.param_name_list[i])[it.multi_index[i]]})
            self.tune_param_grid[it.multi_index] = settings_dict
            #print(it.multi_index)
            it.iternext()

    def singleton_tune_dict(self):
        if self.tune_param_grid.size!=1:
            raise ValueError("not a singleton")
        else:
            it = numpy.nditer(self.tune_param_grid, flags=["multi_index", "refs_ok"])
            tune_dict = self.tune_param_grid[it.multi_index]
        return(tune_dict)



    def clone(self):
        input_dict = self.input_dict.copy()
        if hasattr(self,"cov"):
            if not getattr(self,"cov")=="adapt":
                copy = getattr(self,"cov")[0].clone()
                input_dict["cov"] = [copy]

        #print(input_dict)

        out = tuneinput_class(input_dict)
        return(out)


