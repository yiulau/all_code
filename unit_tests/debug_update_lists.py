from adapt_util.update_list_util import f,fmf,fmsmf,m,msm,s,fsf

out = fmsmf(tune_l=1000)

print(out[0])
print(out[1])
print(out[2])

out = fmf(tune_l=1000)
print(out[0])
print(out[1])
print(out[2])

out = fsf(tune_l=1000)
print(out[0])
print(out[1])
print(out[2])

out = s(tune_l=1000)

print(out[0])
print(out[1])
print(out[2])




