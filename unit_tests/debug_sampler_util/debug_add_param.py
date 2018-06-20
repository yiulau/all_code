from distributions.neural_nets.fc_V_hierarchical import V_fc_test_hyper
import torch.nn as nn
import torch
v_obj = V_fc_test_hyper()


print(len(list(v_obj.parameters())))

obj  = nn.Parameter(torch.zeros(1),requires_grad=True)
setattr(v_obj,"test",obj)

print(len(list(v_obj.parameters())))