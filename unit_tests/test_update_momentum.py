import numpy,os
import pickle
import torch

precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)
seedid = 30
numpy.random.seed(seedid)
torch.manual_seed(seedid)

address = os.environ["PYTHONPATH"] + "/experiments/correctdist_experiments/result_from_long_chain.pkl"
correct = pickle.load(open(address, 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()



#print(correct_diag_cov.shape)
#exit()

print(correct_mean)
print(correct_cov)
print(correct_diag_cov)

torch_cov = torch.from_numpy(correct_cov).type(precision_type)


L = torch.potrf(torch_cov, upper=False)
recomposed_cov = L.mm(torch.t(L))

diff_recomposed = ((recomposed_cov - torch_cov)*(recomposed_cov - torch_cov)).sum()
print("diff recomposed {}".format(diff_recomposed))

Lt_inv = torch.inverse(L.t())
L_inv = torch.inverse(L)
Cov_inv = torch.inverse(torch_cov)
Cov_inv_recomposed = Lt_inv.mm(L_inv)
Cov_inv_L = torch.potrf(Cov_inv,upper=False)
Cov_inv_recomposed_2 = Cov_inv_L.mm(Cov_inv_L.t())


diff_inv_recomposed = ((Cov_inv - Cov_inv_recomposed)*(Cov_inv - Cov_inv_recomposed)).sum()
print("diff recomposed inv {}".format(diff_inv_recomposed))

diff_inv_recomposed_2 = ((Cov_inv - Cov_inv_recomposed_2)*(Cov_inv - Cov_inv_recomposed_2)).sum()
print("diff recomposed inv2 {}".format(diff_inv_recomposed_2))

dim = L_inv.shape[0]
num_sam = 120000
samples = torch.zeros(num_sam,dim)
for i in range(num_sam):
    samples[i,:] = torch.mv(L_inv.t(), torch.randn(dim))

store = samples.numpy()
sample_inv_cov = numpy.cov(store,rowvar=False)

diff_sample_cov = ((sample_inv_cov - Cov_inv)*(sample_inv_cov - Cov_inv)).sum()
print("diff sample vs expected {}".format(diff_sample_cov))

#print(Cov_inv)
#print(sample_inv_cov)

