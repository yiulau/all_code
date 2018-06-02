import torch,numpy

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






