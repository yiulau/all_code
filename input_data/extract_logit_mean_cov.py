import os, pystan, pickle, numpy
from input_data.convert_data_to_dict import get_data_dict
def compute_store_logit_mean_cov(data_name,standardize_predictor):

    dataset = get_data_dict(data_name,standardize_predictor)

    X = dataset["input"]
    y = dataset["target"]
    # y = y.astype(int)
    N = X.shape[0]
    p = X.shape[1]

    data_dict = {"X": X, "y": y, "N": N, "p": p}
    address = os.environ["PYTHONPATH"] + "/stan_code/log_reg_density.stan"
    dump_address = os.environ["PYTHONPATH"] + "/stan_code/log_reg_density_model.pkl"

    if os.path.isfile(dump_address):
        recompile = False
    else:
        recompile = True
    if recompile:
        mod = pystan.StanModel(file=address)
        with open(dump_address, 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open(dump_address, 'rb'))

    fit = mod.sampling(data=data_dict, seed=20)
    print(fit)
    la = fit.extract(permuted=True)
    out = la["beta"]
    mean = numpy.mean(out, axis=0)
    cov = numpy.cov(out, rowvar=False)

    if standardize_predictor:
        qualify = "standardized"
    else:
        qualify = "not_standardized"
    result_dump_address = os.environ["PYTHONPATH"]+"/input_data/"+"_"+data_name+qualify+".npz"
    #result = {"mean":mean,"cov":cov}
    # with open(result_dump_address, 'wb') as f:
    #     pickle.dump(result, f)
    numpy.savez(result_dump_address,mean=mean,cov=cov,allow_pickle=False)
    return()

def extract_logit_mean_cov(data_name,standardize_predictor):
    if standardize_predictor:
        qualify = "standardized"
    else:
        qualify = "not_standardized"
    address = os.environ["PYTHONPATH"]+"/input_data/"+"_"+data_name+qualify+".npz"

    if os.path.isfile(address):
        result = numpy.load(address)
    else:
        raise ValueError("pickled mean and cov does not exist. run compute mean cov")

    return(result)
