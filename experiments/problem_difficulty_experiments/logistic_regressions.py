import pystan,pickle,os,numpy
from input_data.convert_data_to_dict import get_data_dict
stan_sampling = True
if stan_sampling:
    recompile = False
    if recompile:
        address = os.environ["PYTHONPATH"] + "/stan_code/alt_log_reg.stan"
        mod = pystan.StanModel(file=address)
        with open('model.pkl', 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open('model.pkl', 'rb'))


full_data = get_data_dict("pima_indian")
y = full_data["target"].astype(numpy.int64)
X = full_data["input"]
N = X.shape[0]
p = X.shape[1]
data = {"y":y,"X":X,"N":N,"p":p}
fit = mod.sampling(data=data, seed=20)
print(fit)
correct_samples = fit.extract(permuted=True)["beta"]

correct_mean = numpy.mean(correct_samples,axis=0)

correct_cov = numpy.cov(correct_samples,rowvar=False)

sd_vec = numpy.sqrt(numpy.diagonal(correct_cov))

print(sd_vec)

print(max(sd_vec)/min(sd_vec))