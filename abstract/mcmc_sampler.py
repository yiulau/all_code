import time
#from multiprocessing import Pool
#import pathos.multiprocessing as multiprocessing
from pathos.multiprocessing import ProcessPool
#import multiprocessing
import pickle,copy
from abstract.deprecated_code.abstract_genleapfrog_ult_util import *
from abstract.abstract_leapfrog_util import *
from abstract.integrator import sampler_one_step
from adapt_util.adapter_class import adapter_class
from adapt_util.tune_param_classes.tune_param_setting_util import default_adapter_setting
from adapt_util.tune_param_classes.tuning_param_obj import tuning_param_states
from general_util.memory_util import to_pickle_memory
from abstract.abstract_class_point import point
from adapt_util.tune_param_classes.tune_param_class import tune_param_objs_creator
from general_util.pytorch_util import convert_q_point_list
import numpy,torch
from post_processing.get_diagnostics import energy_diagnostics,process_diagnostics
# number of samples
# thinning
# warm up
# initialize at specific point
# number of chains
# parallel computing
# gpu


# delta is a slow parameter like cov and cov_diag, need to be upgraded slowly. consuming signifcant number of samples at
# each update
# integration time t is also a slow parameter, because diagnostics (ESS) for its performance can only be calculated by looking
# at a number of samples

def mcmc_sampler_settings_dict(mcmc_id,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=None,warmup_per_chain=None,is_float=False,isstore_to_disk=False,same_init=False,allow_restart=True,max_num_restarts=10,restart_end_buffer=100,seed=None):
        if seed is None:
            seed = round(numpy.random.uniform(1,1e3))
        else:
            assert seed > 0
            seed = seed

        # mcmc_id should be a dictionary
        out = {}
        if warmup_per_chain is None:
            warmup_per_chain = round(samples_per_chain/2)
        if tune_l_per_chain is None:
            tune_l_per_chain = round(warmup_per_chain/2)
        out.update({"num_chains":num_chains,"num_cpu":num_cpu,"thin":thin,"warmup_per_chain":warmup_per_chain})
        out.update({"is_float":is_float,"isstore_to_disk":isstore_to_disk,"mcmc_id":mcmc_id})
        out.update({"num_samples_per_chain":samples_per_chain,"same_init":same_init,"tune_l_per_chain":tune_l_per_chain})
        out.update({"allow_restart":allow_restart,"max_num_restarts":max_num_restarts,"restart_end_buffer":restart_end_buffer})
        out.update({"seed":seed})
        return(out)

class mcmc_sampler(object):
    # may share experiment id with other sampling_objs
    # tune_settings_dict one object for all 4 (num = numchains) chains
    def __init__(self,tune_dict,mcmc_settings_dict,tune_settings_dict=None,experiment_obj=None,init_q_point_list=None,adapter_setting=None):
        #for name, val in mcmc_meta_obj.__dict__.items():
         #   setattr(self,name,val)
        self.chains_ready = False
        self.tune_dict = tune_dict
        self.v_fun = self.tune_dict["v_fun"]
        self.tune_settings_dict = tune_settings_dict
        self.mcmc_settings_dict = mcmc_settings_dict
        self.num_chains = self.mcmc_settings_dict["num_chains"]
        self.same_init = self.mcmc_settings_dict["same_init"]
        self.store_chains = numpy.empty(self.num_chains, object)
        if self.mcmc_settings_dict["is_float"]:
            self.precision_type = "torch.FloatTensor"
        else:
            self.precision_type = "torch.DoubleTensor"
        torch.set_default_tensor_type(self.precision_type)
        self.seed = self.mcmc_settings_dict["seed"]
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if init_q_point_list is None:
            self.init_q_point_list = default_init_q_point_list(v_fun=self.tune_dict["v_fun"],
                                                               num_chains=self.num_chains,
                                                               same_init=self.same_init,precision_type=self.precision_type,seed=self.seed)
        else:
            self.init_q_point_list= init_q_point_list
            convert_q_point_list(q_point_list=init_q_point_list,precision_type=self.precision_type)
        #print(mcmc_settings_dict.keys())
        #exit()
        #self.Ham = Ham
        if not experiment_obj is None:
            self.experiment_id = experiment_obj.id
        else:
            self.experiment_id = None
        if adapter_setting is None:
            self.adapter_setting = default_adapter_setting()
        else:
            self.adapter_setting = adapter_setting
        if not hasattr(self,"sampler_id"):
            self.sampler_id = 0

        self.metadata = sampler_metadata(mcmc_sampler_obj=self)

        self.num_samples_per_chain = self.mcmc_settings_dict["num_samples_per_chain"]
        self.isstore_to_disk = self.mcmc_settings_dict["isstore_to_disk"]
        self.warmup_per_chain = self.mcmc_settings_dict["warmup_per_chain"]
        self.tune_l_per_chain = self.mcmc_settings_dict["tune_l_per_chain"]

        self.allow_restart = self.mcmc_settings_dict["allow_restart"]
        if self.allow_restart:
            self.max_num_restarts = self.mcmc_settings_dict["max_num_restarts"]
            assert self.mcmc_settings_dict["restart_end_buffer"] < self.warmup_per_chain
            self.restart_end_buffer = self.mcmc_settings_dict["restart_end_buffer"]
        else:
            self.max_num_restarts = 0
            self.restart_end_buffer = 0
        self.is_float = self.mcmc_settings_dict["is_float"]
    def prepare_chains(self):
        # same init = True means the chains will have the same initq, does not affect tuning parameters
        #initialization_obj = initialization(same_init)

        for i in range(self.num_chains):
            #if same_init:
             #   initialization_obj = initialization()
            #print(self.experiment_id)
            #print(self.init_q_point_list)
            #print(self.init_q_point_list[i].flattened_tensor)

            this_chain_setting = one_chain_settings_dict(sampler_id=self.sampler_id,chain_id=i,
                                                         experiment_id=self.experiment_id,
                                                         num_samples=self.num_samples_per_chain,
                                                         warm_up=self.warmup_per_chain,tune_l=self.tune_l_per_chain,
                                                         allow_restart=self.allow_restart,
                                                         max_num_restarts=self.max_num_restarts,is_float=self.is_float,
                                                         restart_end_buff=self.restart_end_buffer)
            #this_tune_dict = self.tune_dict
            this_chain_obj = one_chain_obj(init_point=self.init_q_point_list[i],
                                           tune_dict=copy.deepcopy(self.tune_dict),chain_setting=copy.deepcopy(this_chain_setting),
                                           tune_settings_dict=copy.deepcopy(self.tune_settings_dict),
                                           adapter_setting=copy.deepcopy(self.adapter_setting))
            this_chain_obj.prepare_this_chain()
            this_chain_obj.isstore_to_disk = self.isstore_to_disk
            this_chain_obj.warmup_per_chain = self.warmup_per_chain
            #this_chain_obj.tune_settings_dict = self.tune_settings_dict.copy()
            #print(i)
            self.store_chains[i] = {"chain_obj":this_chain_obj}
        self.chains_ready = True
        #print(self.store_chains)

    def run_chain(self,chain_id):
        #if not self.chains_ready:
        #self.prepare_chains()
            #raise ValueError("run self.prepare_chains() firstf")
        #print("yes")
        #exit()
        #numpy.random.seed(self.seed)
        #seed = numpy.random.uniform(1, 1e3)*self.seed
        seed = self.seed
        torch.manual_seed(round(seed *(chain_id + 1)))
        numpy.random.seed(round(seed * (chain_id + 1)))
        (self.store_chains[chain_id]["chain_obj"]).run()
        #output = self.store_chains[chain_id]["chain_obj"].store_samples
        return()

    def parallel_sampling(self):
        self.num_cpu = self.mcmc_settings_dict["num_cpu"]
        new_mcmc_settings_dict = copy.deepcopy(self.mcmc_settings_dict)
        new_mcmc_settings_dict.update({"num_chains": 1, "num_cpu": 1})
        def run_parallel_chain(chain_id):
            #numpy.random.seed(self.seed)
            #seed = numpy.random.uniform(1, 1e3)*self.seed
            seed = self.seed
            seed = round((seed*(chain_id+1)))
            new_mcmc_settings_dict.update({"seed":seed})
            torch.manual_seed(round(seed*(chain_id+1)))
            numpy.random.seed(round(seed*(chain_id+1)))
            temp_mcmc_sampler = mcmc_sampler(tune_dict=self.tune_dict, mcmc_settings_dict=new_mcmc_settings_dict,
                                             tune_settings_dict=self.tune_settings_dict,
                                             adapter_setting=self.adapter_setting)
            temp_mcmc_sampler.start_sampling()
            return(temp_mcmc_sampler)
        with ProcessPool(nodes=self.num_cpu) as pool:
            result_parallel = pool.map(run_parallel_chain,list(range(self.num_chains)))

        for i in range(self.num_chains):
            self.store_chains[i] = result_parallel[i].store_chains[0]

        return()
    def start_sampling(self):
        self.num_cpu = self.mcmc_settings_dict["num_cpu"]

        if self.num_cpu>1:
            self.parallel_sampling()
            #out = result_parallel
        else:
            if not self.chains_ready:
                self.prepare_chains()
            result_sequential = []
            #print("yes")
            #print(self.num_chains)
            #exit()
            for i in range(self.num_chains):
                #print(i)
                #self.run(0)
                result_sequential.append(self.run_chain(i))
                #experiment = one_chain_sampling(self.precision, self.initialization, self.sampler_one_step, self.adapter)
            #out = result_sequential
        return()

    def pre_sampling_diagnostics(self,test_run_chain_length=15):
        # get
        # 1 estimated total volume
        # 2 estimated  computing time (per chain )
        # 3 estimated total computing time (serial)
        # 4 estimated total coputing time (if parallel, given number of agents)
        # 5 ave number of time per leapfrog
        #initialization_obj = initialization(chain_length = test_run_chain_length)
        #temp_experiment = self.experiment.clone(initialization_obj)
        #temp_sampler =
        #print(self.mcmc_settings_dict)
        #print("yes")
        self.num_samples_per_chain = self.mcmc_settings_dict["num_samples_per_chain"]
        #print(self.store_chains[0]["chain_obj"].adapter.adapter_meta.tune)
        #exit()
        #print(self.store_chains[0])
        #exit()
        if self.store_chains[0]["chain_obj"].adapter.adapter_meta.tune:

            temp_mcmc_meta = self.mcmc_settings_dict.copy()
            temp_mcmc_meta["num_chains"]=1
            temp_mcmc_meta["num_cpu"] = 1
            temp_mcmc_meta["isstore_to_disk"] = False
            temp_mcmc_meta["num_samples_per_chain"] = temp_mcmc_meta["warmup_per_chain"] + test_run_chain_length
        else:
            temp_mcmc_meta = self.mcmc_settings_dict.copy()
            temp_mcmc_meta["num_chains"] = 1
            temp_mcmc_meta["num_cpu"] = 1
            temp_mcmc_meta["isstore_to_disk"] = False
            temp_mcmc_meta["num_samples_per_chain"]=test_run_chain_length

        temp_sampler = mcmc_sampler(tune_dict=self.tune_dict,mcmc_settings_dict=temp_mcmc_meta,
                                    tune_settings_dict=self.tune_settings_dict,
                                    experiment_obj=None, init_q_point_list=None, adapter_setting=self.adapter_setting)


        temp_sampler.start_sampling()
        #exit()
        out = {}
        diagnostics = temp_sampler.sampler_metadata.diagnostics()
        total_warm_up_iter = diagnostics["total_warm_up_iter"]
        total_fixed_tune_iter = diagnostics["total_fixed_tune_iter"]
        total_warm_up_time = diagnostics["total_warm_up_time"]
        total_fixed_tune_time = diagnostics["total_fixed_tune_time"]
        total_num_samples = total_fixed_tune_iter+total_fixed_tune_iter
        total_time = total_warm_up_time+total_fixed_tune_time
        #temp_ave_second_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
        #temp_num_samples = temp_sampler.sampler_metadta.get_num_samples()
        total_size = temp_sampler.sampler_metadata.get_size_mb()
        estimated_total_volume = total_size * (self.num_chains * self.num_samples_per_chain)/total_num_samples
        out.update({"total_volume":estimated_total_volume})

        estimated_compute_wu_seconds_per_chain = total_warm_up_time * self.warmup_per_chain/total_warm_up_iter
        out.update({"warm up seconds_per_chain":estimated_compute_wu_seconds_per_chain})
        fixed_tune_per_chain = self.num_samples_per_chain - self.warmup_per_chain
        estimated_compute_ft_seconds_per_chain = total_fixed_tune_time * fixed_tune_per_chain / total_warm_up_iter
        out.update({"fixed tune seconds_per_chain": estimated_compute_wu_seconds_per_chain})
        estimated_compute_seconds_per_chain  = estimated_compute_wu_seconds_per_chain + estimated_compute_ft_seconds_per_chain
        estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
        out.update({"total_seconds":estimated_compute_seconds})
        estimated_compute_seconds_parallel = estimated_compute_seconds/self.num_cpu
        out.update({"total_seconds with parallel":estimated_compute_seconds_parallel})


        return(out)

    def remove_failed_chains(self):
        assert self.allow_restart
        success_indices = []
        num_restart = []
        for i in range(self.num_chains):
            num_restart.append(self.store_chains[i]["chain_obj"].num_restarts)
            self.metadata.num_restarts[i] = self.store_chains[i]["chain_obj"].num_restarts
            if not self.store_chains[i]["chain_obj"].enough_restarts:
                success_indices.append(i)

        self.metadata.total_num_restarts += sum(num_restart)
        self.metadata.num_chains_removed = self.num_chains - len(success_indices)
        self.store_chains = self.store_chains[success_indices]
        self.num_chains = len(self.store_chains)
        return()
    def get_samples(self,permuted=False):
        # outputs numpy matrix
        if permuted:
            temp_list = []
            for chain in self.store_chains:
                temp_list.append(chain["chain_obj"].get_samples(warmup=self.warmup_per_chain))
            output = temp_list[0]
            if len(temp_list)>0:
                for i in range(1,len(temp_list)):
                    output = numpy.concatenate([output,temp_list[i]],axis=0)
            return(output)
        else:
            chain_shape = self.store_chains[0]["chain_obj"].get_samples(warmup=self.warmup_per_chain).shape
            output = numpy.zeros((self.num_chains,chain_shape[0],chain_shape[1]))
            for i in range(self.num_chains):
                output[i,:,:] = self.store_chains[i]["chain_obj"].get_samples(warmup=self.warmup_per_chain)
            return(output)

    def get_samples_p_diag(self, permuted=True):
        # outputs dict
        output = {"samples":None,"diagnostics":None}
        if permuted:
            temp_list = []
            diag_list = []
            for chain in self.store_chains:
                temp_list.append(chain["chain_obj"].get_samples(warmup=self.warmup_per_chain))
                diag_list.append(chain["chain_obj"].get_diagnostics(warmup=self.warmup_per_chain))
            samples_output = temp_list[0]
            diag_output = diag_list[0]
            if len(temp_list) > 0:
                for i in range(1, len(temp_list)):
                    samples_output = numpy.concatenate([samples_output, temp_list[i]], axis=0)
                    diag_output = diag_output + diag_list[i]
            output.update({"samples":samples_output,"diagnostics":diag_output})
            return (output)
        else:
            chain_shape = self.store_chains[0]["chain_obj"].get_samples(warmup=self.warmup_per_chain).shape
            samples_output = numpy.zeros((self.num_chains, chain_shape[0], chain_shape[1]))
            diag_list = [None]*self.num_chains
            for i in range(self.num_chains):
                samples_output[i, :, :] = self.store_chains[i]["chain_obj"].get_samples(warmup=self.warmup_per_chain)
                diag_list[i] = self.store_chains[i]["chain_obj"].get_diagnostics(warmup=self.warmup_per_chain)
            output.update({"samples":samples_output,"diagnostics":diag_list})
            return (output)

    def get_samples_alt(self, prior_obj_name, permuted=False):
        v_obj = self.v_fun(precision_type=self.precision_type)
        prior_obj = v_obj.dict_parameters[prior_obj_name]
        indices_dict = prior_obj.get_indices_dict()
        if permuted:
            temp_list = []
            for chain in self.store_chains:
                temp_list.append(chain["chain_obj"].get_converted_samples_alt(warmup=self.warmup_per_chain,prior_obj_name=prior_obj_name))
            output = temp_list[0]
            if len(temp_list) > 0:
                for i in range(1, len(temp_list)):
                    output = numpy.concatenate([output, temp_list[i]], axis=0)

        else:
            chain_shape = self.store_chains[0]["chain_obj"].get_converted_samples_alt(warmup=self.warmup_per_chain,prior_obj_name=prior_obj_name).shape
            output = numpy.zeros((self.num_chains, chain_shape[0], chain_shape[1]))
            for i in range(self.num_chains):
                output[i, :, :] = self.store_chains[i]["chain_obj"].get_converted_samples_alt(warmup=self.warmup_per_chain,prior_obj_name=prior_obj_name)

        output_dict = {"samples":output,"indices_dict":indices_dict}
        return (output_dict)

    def get_diagnostics(self,include_warmup=False,permuted=True):
        # outputs dict

        output = {"diagnostics": None,"permuted":permuted}
        if include_warmup:
            warmup = 0
        else:
            warmup = self.warmup_per_chain
        if permuted:
            diag_list = []
            for chain in self.store_chains:
                diag_list.append(chain["chain_obj"].get_diagnostics(warmup=warmup))
            diag_output = diag_list[0]
            if len(diag_list) > 0:
                for i in range(1, len(diag_list)):
                    diag_output = diag_output + diag_list[i]
            output.update({"diagnostics": diag_output})
            return (output)
        else:
            diag_list = [None] * self.num_chains
            for i in range(self.num_chains):
                diag_list[i] = self.store_chains[i]["chain_obj"].get_diagnostics(warmup=warmup)
            output.update({"diagnostics": diag_list})

        return(output)

    def np_diagnostics(self):
        feature_names = ["num_restarts", "num_divergent","hit_max_tree_depth","ave_num_transitions","total_num_transitions","bfmi","lp_ess",
                         "lp_rhat", "difficulty","num_chains_removed"]

        self.remove_failed_chains()
        out = self.get_diagnostics(permuted=False)
        num_restarts = self.metadata.num_restarts
        num_chains_removed = self.metadata.num_chains_removed
        if self.tune_dict["dynamic"]:
            processed_diag = process_diagnostics(out, name_list=["hit_max_tree_depth"])
            hit_max_tree_depth = numpy.squeeze(processed_diag.sum(axis=1))
        else:
            hit_max_tree_depth = 0
        processed_diag = process_diagnostics(out, name_list=["divergent"])
        num_divergent = numpy.squeeze(processed_diag.sum(axis=1))

        processed_diag = process_diagnostics(out, name_list=["num_transitions"])
        total_num_transitions = numpy.sum(processed_diag)
        ave_num_transitions = numpy.squeeze(processed_diag.mean(axis=1))
        energy_summary = energy_diagnostics(diagnostics_obj=out)
        mixed_mcmc_tensor = self.get_samples(permuted=True)
        mcmc_cov = numpy.cov(mixed_mcmc_tensor, rowvar=False)
        mcmc_sd_vec = numpy.sqrt(numpy.diagonal(mcmc_cov))
        difficulty = max(mcmc_sd_vec) / min(mcmc_sd_vec)
        num_id = self.num_chains
        output = numpy.zeros((num_id, len(feature_names)))

        output[:, 0] = num_restarts
        output[:, 1] = num_divergent
        output[:, 2] = hit_max_tree_depth
        output[:, 3] = ave_num_transitions
        output[:, 4] = total_num_transitions
        output[:, 5] = energy_summary["bfmi_list"]
        output[:, 6] = energy_summary["ess"]
        output[:, 7] = energy_summary["rhat"]
        output[:, 8] = difficulty
        output[:, 9] = num_chains_removed
        return(output,feature_names)




# metadata only matters after sampling has started
class sampler_metadata(object):
    def __init__(self,mcmc_sampler_obj):
        self.mcmc_sampler_obj = mcmc_sampler_obj
        self.total_time = 0
        self.num_chains_removed = 0
        self.num_restarts = [0]*mcmc_sampler_obj.num_chains
        self.total_num_restarts = 0
    def store_to_disk(self):
        if self.store_address is None:
            self.store_address = "mcmc_sampler.pkl"
        with open(self.store_address, 'wb') as f:
            pickle.dump(self, f)

    def start_time(self):
        self.start_time = time.time()
    def end_time(self):
        self.total_time += self.start_time - time.time()

    def diagnostics(self):
        tune_l_time_list = []
        fixed_tune_time_list = []
        num_tune_l_iter_list = []
        num_fixed_tune_iter_list = []
        for chain in self.mcmc_sampler_obj.store_chains:
            chain_obj = chain["chain_obj"]
            tune_l_time_this_chain_list = [0]*chain_obj.chain_setting["num_samples"]
            fixed_tune_time_this_chain_list = [0]*chain_obj.chain_setting["num_samples"]
            num_tune_l_iter_this_chain = chain_obj.adapter.adapter_setting["tune_l"]
            num_fixed_tune_this_chain = chain_obj.chain_setting["num_samples"] - num_tune_l_iter_this_chain
            for i in range(num_tune_l_iter_this_chain):
                tune_l_time_this_chain_list[i] = chain_obj.stores_samples[i]["log_obj"].time_since_creation
            for i in range(num_tune_l_iter_this_chain, num_tune_l_iter_this_chain + num_fixed_tune_this_chain):
                fixed_tune_time_this_chain_list[i] = chain_obj.stores_samples[i]["log_obj"].time_since_creation

            tune_l_time_list.append(tune_l_time_this_chain_list)
            fixed_tune_time_list.append(fixed_tune_time_this_chain_list)
            num_tune_l_iter_list.append(num_tune_l_iter_this_chain)
            num_fixed_tune_iter_list.append(num_fixed_tune_this_chain)

        total_tune_l_time = 0
        total_fixed_tune_time = 0
        total_tune_l_iter = 0
        total_fixed_tune_iter = 0
        for i in range(self.mcmc_sampler_obj.num_chains):
            total_tune_l_time += sum(tune_l_time_list[i])
            total_fixed_tune_time += sum(fixed_tune_time_list[i])
            total_tune_l_iter += num_tune_l_iter_list[i]
            total_fixed_tune_iter +=num_fixed_tune_iter_list[i]

        out = {"total_tune_l_time":total_tune_l_time,"total_fixed_tune_time":total_fixed_tune_time}
        out.update({"total_tune_l_iter":total_tune_l_iter,"total_fixed_tune_iter":total_fixed_tune_iter})
        out.update({"tune_l_time_list":tune_l_time_list,"fixed_tune_time_list":fixed_tune_time_list})
        return(out)
    def get_num_samples(self):
        sum = 0
        for i in range(self.mcmc_sampler_obj.num_chains):
            sum+= len(self.mcmc_sampler_obj.store_chains[i].stores_samples)
        return(sum)

    def get_size_mb(self):
        # save to disk. measure volume, then remove stored copy
        # with open("temp_sampler_volume.pkl", 'wb') as f:
        #     pickle.dump(self.mcmc_sampler_obj, f)
        # size = os.path.getsize("./temp_sampler_volume.pkl") / (1024. * 1024)
        # os.remove("./temp_sampler_volume.pkl")
        size = to_pickle_memory(self)
        return(size)




#par_type_dict= {"epsilon":"fast","evolve_L":"medium","evolve_t":"medium","alpha":"medium","xhmc_delta":"medium","diag_cov":"slow","cov":"slow"}

# unique for individual sampler
#tune_method_dict = {"epsilon":"opt","evolve_t":"opt"}

class one_chain_sample_meta(object):
    def __init__(self,one_chain_obj):
        self.num_restart = 0

    def load(self,sample_meta):
        pass

# want it so that sampler_one_step only has inputq and tuning paramater
# supply tuning parameter with a dictionary
# fixed_tune dict stores name and val of tuning paramter that stays the same throughout the entire chain
# tune dict stores two things : param tuned by pyopt, parm tuned by dual averaging
# always start tuning ep first

# tune_dict for each chain should be independent
class one_chain_obj(object):
    def __init__(self,init_point,tune_dict,chain_setting,
                 tune_settings_dict,adapter_setting=None,sample_meta=None):
        self.chain_setting = chain_setting
        if self.chain_setting["is_float"]:
            self.precision_type = "torch.FloatTensor"
        else:
            self.precision_type = "torch.DoubleTensor"
        self.store_samples = []
        self.chain_ready = False
        self.v_fun = tune_dict["v_fun"]
        self.tune_settings_dict = tune_settings_dict
        self.store_log_obj = []
        if self.chain_setting["allow_restart"]:
            self.num_restarts = 0
            self.enough_restarts = False


        #print(chain_setting.keys())
        #exit()


        #print(self.sampling_metadata.__dict__)
        #for param,val in self.sampling_metadata.__dict__.items():
        #    setattr(self,param,val)
        self.tune_dict = tune_dict


        #if tuning_param_settings is None:
        #    self.tuning_param_settings = tuning_param_settings(tune_dict)
        #else:
        #    self.tuning_param_settings = tuning_param_settings
        #print(adapter_setting is None)
        #exit()
        if adapter_setting is None:
            self.adapter = adapter_class(one_chain_obj=self)
        else:
            self.adapter = adapter_class(one_chain_obj=self,adapter_setting=adapter_setting)


        self.tune_param_objs_dict = tune_param_objs_creator(tune_dict=tune_dict,adapter_obj=self.adapter,
                                                           tune_settings_dict=tune_settings_dict)


        #print(self.tune_param_objs_dict)
        #exit()
        self.tuning_param_states = tuning_param_states(adapter=self.adapter,param_objs_dict=self.tune_param_objs_dict,tune_settings_dict=tune_settings_dict)
        self.adapter.tuning_param_states = self.tuning_param_states
        #print(self.tuning_param_states)
        #exit()
        #self.adapter.tuning_param_states = self.tuning_param_states
        #if initialization_obj.tune_param_obj_dict is None:
        #    self.tune_param_obj_dict = tune_params_obj_creator(tune_dict,self.tuning_param_settings)
        #else:
        #    self.tune_param_obj_dict = initialization_obj.tune_param_obj_dict
        #self.sampler_one_step = sampler_one_step(self.tune_dict)
        #print(self.tune_param_objs_dict["epsilon"].get_val())
        #print(self.tune_param_objs_dict["evolve_L"].get_val())
        #exit()
        #self.sampler_one_step = sampler_one_step(self.tune_param_objs_dict,init_point)
        self.sampler_one_step = sampler_one_step(tune_param_objs_dict=self.tune_param_objs_dict,init_point=init_point,
                                                 tune_dict=tune_dict)
        self.init_obj = initialization(V_obj=self.sampler_one_step.v_obj,q_point=init_point)
        if "epsilon" in self.tune_param_objs_dict:
            ep_obj = self.tune_param_objs_dict["epsilon"]
            ep_obj.Ham = self.sampler_one_step.Ham

        if "dense_cov" in self.tune_param_objs_dict  or "diag_cov" in self.tune_param_objs_dict:
            if "dense_cov" in self.tune_param_objs_dict:
                cov_obj = self.tune_param_objs_dict["dense_cov"]
            else:
                cov_obj = self.tune_param_objs_dict["diag_cov"]
            cov_obj.Ham = self.sampler_one_step.Ham
        #exit()
        if sample_meta is None:
            self.sample_meta = one_chain_sample_meta(self)
        else:
            self.sample_meta = sample_meta

    def adapt(self,sample_obj):
        # if self.adapter is none, do nothing
        self.adapter.update(sample_obj)
        #self.sampler_one_step = out.sampler_one_step

    def store_to_disk(self):
        if self.store_address is None:
            self.store_address = "chain.mkl"
        with open(self.store_address, 'wb') as f:
            pickle.dump(self, f)

    def prepare_this_chain(self):
        # initiate tuning parameters if they are tuned automatically
        self.log_obj = log_class()
        self.sampler_one_step.log_obj = self.log_obj
        temp_dict = {}
        for name,obj in self.tune_param_objs_dict.items():
            priority = obj.update_priority
            temp_dict.update({priority:obj})
        temp_tuple = sorted(temp_dict.items())
        for priority,obj in temp_tuple:
                obj.initialize_tuning_param()
        self.chain_ready = True

    def run(self):
        #self.initialization_obj.initialize()
        #print(self.warmup_per_chain)
        keep_going = True

        if not self.chain_ready:
            raise ValueError("need to run prepare this chain")
        cur = self.chain_setting["thin"]
        for counter in range(self.chain_setting["num_samples"]):
            if not keep_going:
                break
        #for counter in range(5):
            cur -= 1
            if not cur>0.1:
                keep = True
                cur = self.chain_setting["thin"]
            else:
                keep = False

            #print(self.tune_param_objs_dict["epsilon"].get_val())
            out = self.sampler_one_step.run()

            #print(out.flattened_tensor)
            #exit()
            #self.adapter.log_obj = self.log_obj
            sample_dict = {"q":out,"iter":counter,"log":self.log_obj.snapshot()}
            self.store_log_obj.append(sample_dict["log"])
            if keep:
                self.add_sample(sample_dict=sample_dict)
                if self.is_to_disk_now(counter):
                    self.store_to_disk()
            if counter < self.chain_setting["tune_l"]:#+1:
                #out.iter = counter
                self.adapt(sample_dict)
                # if self.chain_setting["allow_restart"]:
                #     if out["restart"]:
                #         self.restart(out)

            elif counter == self.chain_setting["warm_up"]:
                if self.chain_setting["allow_restart"]:
                    accumulate_accept_rate = 0

                    for i in range(counter-self.chain_setting["restart_end_buff"],counter):
                        accumulate_accept_rate += self.store_log_obj[i]["accept_rate"]
                    accumulate_accept_rate = accumulate_accept_rate/100

                    if accumulate_accept_rate < 0.1:
                        # if self.num_restarts >5:
                        #     print(accumulate_accept_rate)
                        #     exit()
                        #exit()
                        if self.enough_restarts:
                            break
                        self.restart()
                        keep_going = False




            #print("tune_l is {}".format(self.chain_setting["tune_l"]))
            #print(out)
            print(out.flattened_tensor)
            print("iter is {}".format(counter))
            ep= self.tune_param_objs_dict["epsilon"].get_val()
            print("epsilon val {}".format(self.tune_param_objs_dict["epsilon"].get_val()))
            if "evolve_L" in self.tune_param_objs_dict:
                print("evolve_L val {}".format(self.tune_param_objs_dict["evolve_L"].get_val()))
            if "evolve_t" in self.tune_param_objs_dict:
                t = self.tune_param_objs_dict["evolve_t"].get_val()
                print("evolve_t val {}".format(self.tune_param_objs_dict["evolve_t"].get_val()))
                print("effective L {}".format(round(t/ep)))
            print("number of leapfrog steps {}".format(self.log_obj.store["num_transitions"]))
            print("accept_rate {}".format(self.log_obj.store["accept_rate"]))
            print("accepted {}".format(self.log_obj.store["accepted"]))
            print("divergent is {}".format(self.log_obj.store["divergent"]))
            print("H is {}".format(self.log_obj.store["prop_H"]))
            print("log_post is {}".format(self.log_obj.store["log_post"]))

        return()
    def add_sample(self,sample_dict):
        #print(self.log_obj.store)
        self.store_samples.append(sample_dict)

    def is_to_disk_now(self,counter):
        return(False)

    def get_samples(self,warmup=None):
        if warmup is None:
            warmup = self.chain_setting["warmup"]
        num_out = len(self.store_samples) - warmup
        assert num_out >=1
        store_torch_matrix = torch.zeros(num_out,len(self.store_samples[0]["q"].flattened_tensor))
        # load into tensor matrix
        for i in range(num_out):
            store_torch_matrix[i,:].copy_(self.store_samples[i+warmup]["q"].flattened_tensor)

        store_matrix = store_torch_matrix.numpy()
        return(store_matrix)

    def get_converted_samples_alt(self,prior_obj_name,warmup=None):
        v_obj = self.v_fun(precision_type=self.precision_type)
        assert prior_obj_name in v_obj.dict_parameters
        prior_obj = v_obj.dict_parameters[prior_obj_name]
        dim = len(prior_obj.get_all_param_flattened())
        if warmup is None:
            warmup = self.chain_setting["warmup"]
        num_out = len(self.store_samples) - warmup
        assert num_out >=1
        store_torch_matrix = torch.zeros(num_out,dim)
        # load into tensor matrix
        for i in range(num_out):
            v_obj.flattened_tensor.copy_(self.store_samples[i+warmup]["q"].flattened_tensor)
            v_obj.load_flattened_tensor_to_param()
            store_torch_matrix[i,:] = prior_obj.get_all_param_flattened()
        return(store_torch_matrix)



    def get_diagnostics(self,warmup=None):
        if warmup is None:
            warmup = self.chain_setting["warmup"]
        num_out = len(self.store_samples) - warmup
        assert num_out >=1
        out = [None]*num_out
        for j in range(num_out):
            #print(j+warmup)
            out[j] = self.store_samples[j+warmup]["log"]
        return(out)


    def restart(self):
         # return to start state
         # load samplemeta
         # erase saved samples (if any)
         # run
         self.store_samples = []
         self.store_log_obj = []
         self.prepare_this_chain()
         self.num_restarts+=1
         if self.num_restarts > self.chain_setting["max_num_restarts"]:
             self.enough_restarts = True
         if not self.enough_restarts:
             self.init_obj.initialize()
             self.run()

         return()

# log_obj should keep information about dual_obj, and information about tuning parameters
# created at the start of each transition. discarded at the end of each transition
class log_class(object):
    def __init__(self):
        self.store = {}
        self.start_time = time.time()
    def snapshot(self):
        store_dict = copy.deepcopy(self.store)
        store_dict.update({"time_since_start":self.get_time_since_creation()})
        return(store_dict)

    def get_time_since_creation(self):
        return(time.time()-self.start_time)




class initialization(object):
    # should contain a point object
    def __init__(self,V_obj,q_point=None):
        self.V_obj = V_obj
        if not q_point is None:
            V_obj.load_point(q_point)
        else:
            self.initialize()

    def initialize(self):
        self.V_obj.flattened_tensor.copy_(torch.randn(len(self.V_obj.flattened_tensor))*1.41)
        self.V_obj.load_flattened_tensor_to_param()
        return()


def one_chain_settings_dict(sampler_id,chain_id,num_samples=10,thin=1,experiment_id=None,tune_l=5,warm_up=5,allow_restart=True,max_num_restarts=3,is_float=False,restart_end_buff=1):

        # one for every chain. in everything sampling object, in every experiment
        # parallel chains sampling from the same distribution shares sampling_obj
        # e.g. 4 chains sampling from iid normal
        # different chains sampling in the context of the same experiment shares experiment_obj
        # e.g. HMC(ep,t) sampling from a model with different values of (ep,t) on a grid
        # thin an integer >=0 skips
        # period for saving samples
        out = {"experiment_id":experiment_id,"sampler_id":sampler_id,"chain_id":chain_id}
        out.update({"num_samples":num_samples,"thin":thin,"tune_l":tune_l,"warm_up":warm_up})
        out.update({"allow_restart":allow_restart,"max_num_restarts":max_num_restarts,"is_float":is_float,"restart_end_buff":restart_end_buff})
        return(out)






def default_init_q_point_list(v_fun,num_chains,same_init=False,precision_type="torch.DoubleTensor",seed=None):
    v_obj = v_fun(precision_type=precision_type)
    init_q_point_list = [None]*num_chains
    assert not seed is None
    torch.manual_seed(seed)
    if same_init:
        #print("yes")
        temp_point = point(V=v_obj)
        temp_point.flattened_tensor.copy_(torch.randn(len(temp_point.flattened_tensor)))
        temp_point.load_flatten()
        #print(temp_point.flattened_tensor)
        #print(temp_point)
        #exit()
        for i in range(num_chains):
            init_q_point_list[i] = temp_point.point_clone()
            #print(init_q_point_list[i].flattened_tensor)
    else:
        for i in range(num_chains):
            #temp_point = v_obj.q_point.point_clone()
            temp_point = point(V=v_obj)
            temp_point.flattened_tensor.copy_(torch.randn(len(temp_point.flattened_tensor)))
            temp_point.load_flatten()
            init_q_point_list[i] = temp_point

    return(init_q_point_list)

