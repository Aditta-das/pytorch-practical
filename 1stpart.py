import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 4], [5, 6, 7]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)

x = torch.eye(5, 5)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(0.1, 1, 10)
print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(x)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())

import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)
back_numpy = tensor.numpy()
print(back_numpy)


###############################TENSOR MATH#############################################
x = torch.tensor([1, 2, 4])
y = torch.tensor([9, 8, 7])
z1 = torch.empty(3)
print(torch.add(x, y, out=z1))

z2 = torch.add(x, y)
z = x + y

z = x - y

z = torch.true_divide(x, y)
print(z)

# inplace operation
t = torch.zeros(3)
print(t.add_(x))

z = x.pow(2)
print(z)
z = x ** 2
print(z)

# comparison
z = x > 0
print(z)
z = x < 0

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
print(tensor1)
tensor2 = torch.rand((batch, m, p))
print(tensor2)

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
print(z)

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
z = torch.clamp(x, min=0)
print(z)

###############INDEXING###################
batch_size = 10
features = 25
x = torch.rand(batch_size, features)
print(x[0].shape)
print(x[:, 0].shape)

print(x[2, 0:10])

###############FANCY INDEXING####################
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

####################ADVANCE INDEXING############################
x = torch.arange(10)
print(x[(x < 2) & (x > 5)])
print(x[x.remainder(2) == 0])


print(torch.where(x > 5, x, x * 2))
print(torch.tensor([0,0,1,2,3,4]).unique())
print(x.ndimension())
print(x.numel())

######################RESHAPE#########################
x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)
print(x_3x3)
##################################################################
#torch.view has existed for a long time. It will return a tensor with the new shape. The returned tensor will share the underling data with the original tensor. See the documentation here.

#On the other hand, it seems that torch.reshape has been introduced recently in version 0.4. According to the document, this method will

#    Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.

#It means that torch.reshape may return a copy or a view of the original tensor. You can not count on that to return a view or a copy. According to the developer:

#    if you need a copy use clone() if you need the same storage use view(). The semantics of reshape() are that it may or may not share the storage and you don't know beforehand.

#Another difference is that reshape() can operate on both contiguous and non-contiguous tensor while view() can only operate on contiguous tensor. Also see here about the meaning of contiguous.
######################################################################

y = x_3x3.t()
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)
z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
z = x.squeeze(1)
print(z.shape)


x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())
y = torch.squeeze(x)
print(y.size())