import numpy as np
import pandas as pd
import torch

# a = torch.Tensor([1, 2, 3, 4, 5, 6])
# a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
# a = torch.FloatTensor([1, 2, 3, 4, 5, 6])
# print(a.dtype)
# print(a)
# a = a.type(torch.IntTensor)
# print(a.dtype)
# print(a)

# print(a.size())
# print(a.ndimension())

# a_col = a.view(6, 1)
# a_col = a.view(-1, 1)
# print(a_col)

# numpy_array = np.array([1, 2, 3, 4, 5, 6])
# torch_tensor = torch.from_numpy(numpy_array)
# back_to_numpy = torch_tensor.numpy()

# pandas_series = pd.Series([1, 2, 3, 4, 5, 6])
# pandas_to_torch = torch.from_numpy(pandas_series.values)

# new_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
# print(new_tensor[0])
# print(new_tensor[0].item())

# u = torch.tensor([1.0, 2.0])
# v = torch.tensor([23.0, 3.0])
# z = u**v
# z = torch.dot(u, v)
# print(z)

# a = torch.tensor([1, 2, 3, 4, 5, 6])
# max_b = a.max().item()
# print(max_b)

# print(np.pi)
# print(torch.mean(a))
# print(torch.sin(a))

# b = torch.linspace(0, 3, steps=15)
# print(b)

# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(a.numel())

# my_torch = torch.arange(10)
# print(my_torch)
# # my_torch = my_torch.reshape(2, 5)
# my_torch = my_torch.reshape(2, -1)
# print(my_torch)
# print(my_torch[:, 2:])

# torch_a = torch.tensor([1, 2, 3, 4])
# torch_b = torch.tensor([5, 6, 7, 8])

# print(torch.add(torch_a, torch_b))
# print(torch.sub(torch_a, torch_b))
# print(torch.multiply(torch_a, torch_b))
# print(torch.divide(torch_a, torch_b))
# print(torch.pow(torch_a, torch_b))

# torch_a.add_(torch_b)
# print(torch_a)
