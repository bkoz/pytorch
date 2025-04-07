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

    return device


    
device = check_accelerator()
print(f"Using accelerator device: {device}")

N, D = 3, 4

x = torch.rand((N, D), requires_grad=True).to(device)
y = torch.rand((N, D), requires_grad=True).to(device)
z = torch.rand((N, D), requires_grad=True).to(device)

a = (x * y).to(device)
b = (a + z).to(device)
c = torch.sum(b).to(device)

c.backward()

print(c)

