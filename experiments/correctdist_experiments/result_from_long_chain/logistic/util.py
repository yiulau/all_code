import os, pystan,pickle,numpy

def result_from_long_chain(input_data,data_name,recompile=True):
    stan_sampling = True
    result_name = 'result_from_long_chain_{}.npz'.format(data_name)
    if stan_sampling:
        if recompile:
            address = os.environ["PYTHONPATH"] + "/stan_code/alt_log_reg.stan"
            mod = pystan.StanModel(file=address)
            with open('model.pkl', 'wb') as f:
                pickle.dump(mod, f)
        else:
            mod = pickle.load(open('model.pkl', 'rb'))
    if os.path.isfile(result_name):
        print("file already exists")
    else:
        y_np = input_data["target"]
        y_np = y_np.astype(numpy.int64)
        X_np = input_data["input"]
        num_ob = X_np.shape[0]
        dim = X_np.shape[1]
        data = dict(y=y_np, X=X_np, N=num_ob, p=dim)
        fit = mod.sampling(data=data, seed=1, iter=100000, thin=1)

        correct_samples = fit.extract(permuted=True)["beta"]
        # print(fit)
        correct_mean = numpy.mean(correct_samples, axis=0)
        # print(correct_mean)
        correct_cov = numpy.cov(correct_samples, rowvar=False)

        # print(correct_cov)
        out = {"correct_mean": correct_mean, "correct_cov": correct_cov}
        numpy.savez(result_name,**out)
    return(result_name)