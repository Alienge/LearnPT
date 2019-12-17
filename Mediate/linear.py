import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
m = nn.Linear(20,30)
input = torch.randn(128,20)
output = m(input)
#print(output)
d = torch.Tensor(5, 2)
print(d)
c = Parameter(torch.Tensor(2, 2))
print(c.requires_grad)
print(c)
