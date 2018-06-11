import pystan,pickle,os
from input_data.convert_data_to_dict import get_data_dict
stan_sampling = True

if stan_sampling:
    recompile = False
    if recompile:
        address =  os.environ["PYTHONPATH"] +"/stan_code/linear_regression.stan"
        mod = pystan.StanModel(file=address)
        with open('model.pkl', 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open('model.pkl', 'rb'))


data = get_data_dict("boston")
data_stan = dict(y=data["target"],X=data["input"],N=data["input"].shape[0],K=data["input"].shape[1])

#control_dict = dict({"adapt_engaged":True,"metric":"unit_e",})
fit = mod.sampling(data=data_stan, seed=330)
#fit = mod.sampling(data=data_stan, seed=20,control=control_dict,algorithm="HMC",iter=4000)

print(fit.summary())
exit()
la = fit.extract(permuted=True)
print(la.keys())
ep = la["stepsize__"]

print(len(ep))
print(fit)
print(ep)