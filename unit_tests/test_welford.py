import torch, numpy
from explicit.adapt_util import welford


samples = torch.randn(100,10)
m_ = torch.zeros(10)
m_2 = torch.zeros(10,10)
counter = 0
for i in range(100):
    m_,m_2,counter = welford(samples[i,:],counter,m_,m_2,False)

store = samples.numpy()
empCov = numpy.cov(store, rowvar=False)
empmean = numpy.mean(store, axis=0)

print("real cov {}".format(empCov))
print("real mean {}".format(empmean))

print("welford cov {}".format(m_2/(counter-1)))
print("welford mean {}".format(m_))

welford_mean = m_.numpy()
welford_cov = (m_2/(counter-1)).numpy()
cov_diff = numpy.square((empCov - welford_cov)).sum()
mean_diff = numpy.square((empmean - welford_mean)).sum()
print("cov diff {}".format(cov_diff))
print("mean diff {}".format(mean_diff))