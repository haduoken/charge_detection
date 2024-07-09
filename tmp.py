import torch
from torch import nn

a = torch.tensor([0, 0, 0, 0])

b = torch.rand(4)
c = torch.rand(4)

keep = a == 1

b1 = b[keep]
c1 = c[keep]

print('c1 is empty ? ', c1.numel() == 0)


loss = nn.SmoothL1Loss()

l = loss(b1, c1)

print(l)
