import torch,numpy
def posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type):
    # output store_dist [input_sample i , class probability j , mcmc sample k ] for classification case
    # output store_dist [input_sample i , predicted y , mcmc sample k ]
    assert target_dataset["input"].shape[0] == len(target_dataset["target"])
    permissible_types = ("classification", "regression")
    assert type in permissible_types
    num_mcmc_samples = mcmc_samples.shape[0]
    num_target_samples = target_dataset["input"].shape[0]
    torch_mcmc_samples = torch.from_numpy(mcmc_samples).type(v_nn_obj.flattened_tensor.type())
    if type =="classification":
        test_samples = target_dataset["input"][0:1,:]
        out_prob = v_nn_obj.predict(test_samples)
        num_classes = out_prob.shape[1]
        # code above only to get number of classes
        store_dist = torch.zeros(num_target_samples,num_classes,num_mcmc_samples)
        for i in range(num_mcmc_samples):
            v_nn_obj.flattened_tensor.copy_(torch_mcmc_samples[i,:])
            v_nn_obj.load_flattened_tensor_to_param()
            store_dist[:,:,i].copy_(v_nn_obj.predict(target_dataset["input"]))

    elif type =="regression":
        store_dist = torch.zeros(num_target_samples,num_mcmc_samples)
        for i  in range(num_mcmc_samples):
            v_nn_obj.flattened_tensor.copy_(torch_mcmc_samples[i, :])
            v_nn_obj.load_flattened_tensor_to_param()
            store_dist[:,i].copy_(v_nn_obj.predict(target_dataset["input"]))
    else:
        print(type)
        raise ValueError("unknown type")
    return(store_dist)

def map_prediction(target_dataset,v_nn_obj,mcmc_samples,type,memory_efficient=False):
    # returns map if classification, posterior mean if classification
    # also returns mean,var, mcse, eff for all relevant quantities , class prob for classification, predicted y for
    # regression
    assert type in ("regression","classification")

    num_mcmc_samples = mcmc_samples.shape[0]
    num_target_samples = target_dataset["input"].shape[0]

    if type=="classification":
        if memory_efficient:
            store_prediction = torch.zeros(num_target_samples).type("torch.LongTensor")
            store_prediction_uncertainty = torch.zeros(num_target_samples)
            test_samples = target_dataset["input"][0:1, :]
            out_prob = v_nn_obj.predict(test_samples)
            num_classes = out_prob.shape[1]
            for i in range(num_target_samples):
                # first column stores mean, second column stores var
                store_prob = torch.zeros(num_classes,2)
                debug_store_prob = torch.zeros(num_classes)
                temp = torch.zeros(num_mcmc_samples,num_classes)
                new_sample = target_dataset["input"][i:i + 1, :]
                for j in range(num_mcmc_samples):
                    v_nn_obj.flattened_tensor.copy_(torch.from_numpy(mcmc_samples[j, :]))
                    v_nn_obj.load_flattened_tensor_to_param()
                    out_prob = v_nn_obj.predict(new_sample)
                    temp[j,:].copy_(out_prob[0,:])
                store_prob[:,0] =  temp.mean(dim=0)
                store_prob[:,1] = temp.var(dim=0)
                max_prob,map_class  = torch.max(store_prob[:,0:1],0)
                store_prediction[i] = map_class[0]
                store_prediction_uncertainty[i] = store_prob[map_class[0],1]

        else:
            store_prediction = torch.zeros(num_target_samples).type("torch.LongTensor")
            store_prediction_uncertainty = torch.zeros(num_target_samples)
            post_dist = posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type)
            #print(post_dist)
            #exit()
            post_class_prob_mean = post_dist.mean(dim=2)
            post_class_prob_var = post_dist.var(dim=2)
            map_prob,map_class  = torch.max(post_class_prob_mean,dim=1)
            # print(map_class.shape)
            # print(store_prediction.shape)
            # print(num_mcmc_samples)
            # print(post_class_prob_var.shape)

            store_prediction.copy_(map_class)
            # print(post_class_prob_mean)
            # print(map_class.shape)
            # print(post_class_prob_var.shape)
            # print(post_class_prob_var[0,map_class[0]])

            for i in range(store_prediction.shape[0]):
                store_prediction_uncertainty[i] = post_class_prob_var[i,map_class[i]]
    elif type=="regression":
        if memory_efficient:
            store_prediction = torch.zeros(num_target_samples)
            store_prediction_uncertainty = torch.zeros(num_target_samples)
            for i in range(num_target_samples):
                temp = []*num_mcmc_samples
                for j in range(num_mcmc_samples):
                    new_sample = target_dataset["input"][i:i+1,:]
                    out_predicted = v_nn_obj.predict(new_sample)
                    temp[j] = out_predicted

                store_prediction[i] = numpy.mean(temp)
                store_prediction_uncertainty[i] = numpy.var(temp)
        else:
            store_prediction = torch.zeros(num_target_samples)
            store_prediction_uncertainty = torch.zeros(num_target_samples)
            post_dist = posterior_predictive_dist(target_dataset,v_nn_obj,mcmc_samples,type)
            post_out_mean  = torch.mean(post_dist,dim=1)
            post_out_var = torch.var(post_dist,dim=1)
            store_prediction.copy_(post_out_mean)
            store_prediction_uncertainty.copy_(post_out_var)
    else:
        raise ValueError("unknown type - shouldnt happen")
    return(store_prediction,store_prediction_uncertainty)

def test_error(target_dataset,v_obj,mcmc_samples,type,memory_efficient=False):
    assert type in ("regression","classification")
    predicted,uncertainty = map_prediction(target_dataset=target_dataset,v_nn_obj=v_obj,mcmc_samples=mcmc_samples,
                               type=type,memory_efficient=memory_efficient)
    predicted = predicted.numpy()
    correct_target = target_dataset["target"]
    if type=="classification":
        error = sum(predicted!=correct_target)/len(predicted)
    else:
        error = sum((predicted - correct_target)*(predicted - correct_target))
    return(error,predicted)