from input_data.convert_data_to_dict import get_data_dict
import pystan,os,pickle,numpy
from input_data.extract_logit_mean_cov import compute_store_logit_mean_cov,extract_logit_mean_cov

#compute_store_logit_mean_cov(data_name="pima_indian",standardize_predictor=False)

ot = extract_logit_mean_cov("pima_indian",False)


mean = ot["mean"]
cov = ot["cov"]

diag_sd = numpy.sqrt(numpy.diag(cov))
difficulty = numpy.max(diag_sd)/(numpy.min(diag_sd))
print(difficulty)
#address = "temp.npz"
# numpy.savez(address, mean=mean, cov=cov,allow_pickle=False)

# ot2 = numpy.load("temp.npz")
# mean1 = ot2["mean"]
# cov1 = ot2["cov"]
# print(mean1)
# print(cov1)
exit()



dataset = get_data_dict("pima_indian")

X = dataset["input"]
y = dataset["target"]
#y = y.astype(int)
N = X.shape[0]
p = X.shape[1]

data_dict = {"X":X,"y":y,"N":N,"p":p}

address = os.environ["PYTHONPATH"] +"/stan_code/log_reg_density.stan"

dump_address = os.environ["PYTHONPATH"]+"/stan_code/log_reg_density_model.pkl"

stan_sampling = True
if stan_sampling:
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

#print(fit)

la = fit.extract(permuted=True)

out = la["beta"]

mean = numpy.mean(out,axis=0)

print(mean)

cov = numpy.cov(out,rowvar=False)

print(cov)


