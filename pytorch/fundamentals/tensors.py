"""
PyTorch Tensor Fundamentals
============================
Covers: scalar, vector, matrix, ndim, dtype, shape,
        zeros, ones, and random tensors.
"""

import torch

# -- Scalar ------------------------------------------------------------------
scalar = torch.tensor(7)
print("=== Scalar ===")
print(scalar)               # tensor(7)
print("ndim :", scalar.ndim)    # 0 - no dimensions
print("item :", scalar.item())  # Python int: 7

# -- Vector ------------------------------------------------------------------
vector = torch.tensor([1, 2, 3])
print("\n=== Vector ===")
print(vector)               # tensor([1, 2, 3])
print("ndim  :", vector.ndim)   # 1
print("shape :", vector.shape)  # torch.Size([3])

# -- Matrix ------------------------------------------------------------------
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("\n=== Matrix ===")
print(matrix)
print("ndim  :", matrix.ndim)   # 2
print("shape :", matrix.shape)  # torch.Size([3, 2])

# -- ndim (number of dimensions) ---------------------------------------------
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n=== ndim demo (3-D tensor) ===")
print("ndim  :", tensor_3d.ndim)   # 3
print("shape :", tensor_3d.shape)  # torch.Size([2, 2, 2])

# -- dtype --------------------------------------------------------------------
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor   = torch.tensor([1, 2, 3],        dtype=torch.int64)
print("\n=== dtype ===")
print("float32 tensor :", float_tensor, "| dtype:", float_tensor.dtype)
print("int64   tensor :", int_tensor,   "| dtype:", int_tensor.dtype)

# Cast dtype
casted = float_tensor.to(torch.float16)
print("casted to float16 dtype:", casted.dtype)

# -- shape --------------------------------------------------------------------
shaped = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\n=== shape ===")
print("shape  :", shaped.shape)   # torch.Size([2, 3])
print("size() :", shaped.size())  # torch.Size([2, 3]) - equivalent

# -- Zeros --------------------------------------------------------------------
zeros = torch.zeros(3, 4)          # 3 rows x 4 cols, all 0.0
print("\n=== Zeros ===")
print(zeros)
print("dtype :", zeros.dtype)      # float32 by default

# -- Ones ---------------------------------------------------------------------
ones = torch.ones(2, 3)            # 2 rows x 3 cols, all 1.0
print("\n=== Ones ===")
print(ones)
print("dtype :", ones.dtype)

# -- Random Tensors -----------------------------------------------------------
print("\n=== Random Tensors ===")

# Uniform [0, 1)
rand_uniform = torch.rand(3, 3)
print("rand (uniform [0,1)):\n", rand_uniform)

# Normal distribution (mean=0, std=1)
rand_normal = torch.randn(3, 3)
print("randn (standard normal):\n", rand_normal)

# Random integers in [low, high)
rand_int = torch.randint(low=0, high=10, size=(3, 3))
print("randint [0, 10):\n", rand_int)

# Reproducible randomness via manual seed
torch.manual_seed(42)
reproducible = torch.rand(2, 3)
print("rand with seed 42:\n", reproducible)
