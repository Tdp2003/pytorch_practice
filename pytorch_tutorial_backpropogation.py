import re
import torch
import numpy as np

x = torch.tensor(1.0)
w = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0)

r = x*w
loss = (r-y)**2

print(loss)

loss.backward()
print(w.grad)