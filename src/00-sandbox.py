#
# Basic Pytorch checks
#
import torch
import torch.backends

def check_accelerator():
    
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps" # Apple GPU
    else:
        device = "cpu"

    print(f"Using accelerator device: {device}")

    
check_accelerator()

N, D = 3, 4

x = torch.rand((N, D), requires_grad=True)
y = torch.rand((N, D), requires_grad=True)
z = torch.rand((N, D), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(c)

