#
# Taken from https://www.cs.princeton.edu/courses/archive/fall19/cos484/lectures/pytorch.pdf
#
import torch

print(f'accelerator = {torch.accelerator.is_available()}')

N, D = 3, 4

x = torch.rand((N, D), requires_grad=True)
y = torch.rand((N, D), requires_grad=True)
z = torch.rand((N, D), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(c)

