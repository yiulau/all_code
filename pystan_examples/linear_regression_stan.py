from input_data.convert_data_to_dict import get_data_dict
import numpy,pystan,pickle,os
dataset = get_data_dict("boston",standardize_predictor=False)

N = dataset["input"].shape[0]
K = dataset["input"].shape[1]
X_mean = numpy.mean(dataset["input"],axis=0)

X_cov = numpy.cov(dataset["input"],rowvar=False)

print(X_mean)

print(numpy.diag(X_cov))

data = {"X":dataset["input"],"y":dataset["target"],"N":N,"K":K}

stan_sampling = True
if stan_sampling:
    recompile = False
    if recompile:
        address = os.environ["PYTHONPATH"] +"/stan_code/linear_regression.stan"
        mod = pystan.StanModel(file=address)
        with open('lr_model.pkl', 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open('lr_model.pkl', 'rb'))


#fit = mod.sampling(data=data)
#fit = mod.sampling(data=data,control={"metric":"unit_e"})
#print(fit)

#fit_hmc = mod.sampling(data=data,algorithm="HMC",control={"stepsize":0.1,"int_time":1.,"adapt_engaged":False})

#fit_hmc = mod.sampling(data=data,algorithm="HMC",control={"int_time":1.,"adapt_engaged":True,"metric":"unit_e"})

fit_hmc = mod.sampling(data=data,algorithm="HMC",control={"int_time":1.,"adapt_engaged":True})
print(fit_hmc)