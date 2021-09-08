import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

torch.manual_seed(1)

# lin = nn.Linear(5, 3)
# data = torch.randn(2, 5)
# print(lin(data))

data2 = torch.randn(2, 2)
print(data2)
print(F.relu(data2))

data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())
print(F.log_softmax(data, dim=0))