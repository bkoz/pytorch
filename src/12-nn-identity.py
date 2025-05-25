import torch
import torch.nn as nn

m = nn.Identity(10, unused_argument1=0.1, unused_argument2=False)
input = torch.randn(10)
output = m(input)
print(f'{input=}')
print(f'{output=}')
