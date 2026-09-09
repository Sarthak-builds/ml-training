"""
PyTorch Tensor Operations (up to matmul)
=========================================
Covers: basic arithmetic, aggregation, indexing,
        reshaping, squeezing, stacking, and matrix multiplication.
"""

import torch

torch.manual_seed(42)  # reproducibility

# =============================================================================
# 1. BASIC ARITHMETIC OPERATIONS
# =============================================================================

print("=" * 50)
print("BASIC ARITHMETIC")
print("=" * 50)

t = torch.tensor([1.0, 2.0, 3.0])

print(f"Original : {t}")
print(f"+ 10     : {t + 10}")           # element-wise add
print(f"- 5      : {t - 5}")            # element-wise subtract
print(f"* 2      : {t * 2}")            # element-wise multiply
print(f"/ 2      : {t / 2}")            # element-wise divide
print(f"** 2     : {t ** 2}")           # element-wise power
print(f"% 2      : {t % 2}")            # element-wise modulo

# Equivalent torch functions
print(f"\ntorch.add      : {torch.add(t, 10)}")
print(f"torch.sub      : {torch.sub(t, 5)}")
print(f"torch.mul      : {torch.mul(t, 2)}")
print(f"torch.div      : {torch.div(t, 2)}")

# =============================================================================
# 2. AGGREGATION OPERATIONS
# =============================================================================

print("\n" + "=" * 50)
print("AGGREGATION")
print("=" * 50)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Tensor   : {x}")
print(f"min      : {torch.min(x)}  | x.min()  : {x.min()}")
print(f"max      : {torch.max(x)}  | x.max()  : {x.max()}")
print(f"mean     : {torch.mean(x)} | x.mean() : {x.mean()}")
print(f"sum      : {torch.sum(x)}  | x.sum()  : {x.sum()}")
print(f"std      : {torch.std(x):.4f}")

# argmin / argmax - index of min/max value
print(f"\nargmin   : {torch.argmin(x)} (index of min)")
print(f"argmax   : {torch.argmax(x)} (index of max)")

# Aggregation along a dimension
m = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
print(f"\nMatrix:\n{m}")
print(f"sum dim=0 (col-wise): {m.sum(dim=0)}")   # sum down rows
print(f"sum dim=1 (row-wise): {m.sum(dim=1)}")   # sum across cols
print(f"mean dim=0          : {m.mean(dim=0)}")

# =============================================================================
# 3. INDEXING & SLICING
# =============================================================================

print("\n" + "=" * 50)
print("INDEXING & SLICING")
print("=" * 50)

t = torch.arange(0, 9).reshape(3, 3).float()
print(f"Tensor:\n{t}")

print(f"\nt[0]      : {t[0]}")          # first row
print(f"t[0][1]   : {t[0][1]}")        # row 0, col 1
print(f"t[0, 1]   : {t[0, 1]}")        # same as above
print(f"t[:, 1]   : {t[:, 1]}")        # all rows, col 1
print(f"t[1:, 1:] :\n{t[1:, 1:]}")     # rows 1+, cols 1+

# =============================================================================
# 4. RESHAPING OPERATIONS
# =============================================================================

print("\n" + "=" * 50)
print("RESHAPING")
print("=" * 50)

x = torch.arange(1, 13)                # [1, 2, ..., 12]  shape [12]
print(f"Original shape  : {x.shape}")

reshaped = x.reshape(3, 4)             # reshape to [3, 4]
print(f"reshape(3,4)    : {reshaped.shape}\n{reshaped}")

viewed = x.view(4, 3)                  # view shares memory with original
print(f"view(4,3)       : {viewed.shape}")

flattened = reshaped.flatten()         # collapse all dims to 1-D
print(f"flatten()       : {flattened.shape} | {flattened}")

# =============================================================================
# 5. SQUEEZE & UNSQUEEZE
# =============================================================================

print("\n" + "=" * 50)
print("SQUEEZE & UNSQUEEZE")
print("=" * 50)

t = torch.zeros(1, 3, 1, 2)
print(f"Original shape    : {t.shape}")      # [1, 3, 1, 2]

squeezed = t.squeeze()                        # remove all size-1 dims
print(f"squeeze()         : {squeezed.shape}")  # [3, 2]

squeezed_0 = t.squeeze(0)                     # remove dim 0 only
print(f"squeeze(0)        : {squeezed_0.shape}")# [3, 1, 2]

t2 = torch.tensor([1.0, 2.0, 3.0])           # shape [3]
unsqueezed = t2.unsqueeze(0)                  # add dim at position 0
print(f"\nunsqueeze(0)      : {unsqueezed.shape}")  # [1, 3]
unsqueezed_1 = t2.unsqueeze(1)               # add dim at position 1
print(f"unsqueeze(1)      : {unsqueezed_1.shape}")  # [3, 1]

# =============================================================================
# 6. STACKING & CONCATENATION
# =============================================================================

print("\n" + "=" * 50)
print("STACKING & CONCATENATION")
print("=" * 50)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

stacked = torch.stack([a, b], dim=0)          # new dim: [2, 3]
print(f"stack dim=0 : shape {stacked.shape}\n{stacked}")

stacked_1 = torch.stack([a, b], dim=1)        # new dim: [3, 2]
print(f"stack dim=1 : shape {stacked_1.shape}\n{stacked_1}")

cat_0 = torch.cat([a, b], dim=0)              # concat along dim 0: [6]
print(f"cat dim=0   : shape {cat_0.shape} | {cat_0}")

# =============================================================================
# 7. PERMUTE (transpose dimensions)
# =============================================================================

print("\n" + "=" * 50)
print("PERMUTE")
print("=" * 50)

img = torch.rand(3, 64, 64)                   # C x H x W
print(f"Original (C,H,W) : {img.shape}")

img_permuted = img.permute(1, 2, 0)           # H x W x C  (common for matplotlib)
print(f"permute(1,2,0)   : {img_permuted.shape}")

# =============================================================================
# 8. MATRIX MULTIPLICATION (matmul)
# =============================================================================

print("\n" + "=" * 50)
print("MATRIX MULTIPLICATION (matmul)")
print("=" * 50)

# Rule: (m, n) @ (n, p) -> (m, p)   inner dims must match
A = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0],
                   [5.0, 6.0]])          # shape [3, 2]

B = torch.tensor([[7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]])  # shape [2, 3]

print(f"A shape : {A.shape}")
print(f"B shape : {B.shape}")

# Method 1: torch.matmul
result1 = torch.matmul(A, B)
print(f"\ntorch.matmul(A, B) -> shape {result1.shape}:\n{result1}")

# Method 2: @ operator (identical)
result2 = A @ B
print(f"\nA @ B              -> shape {result2.shape}:\n{result2}")

# Dot product (1-D vectors)
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
dot = torch.dot(v1, v2)                  # 1*4 + 2*5 + 3*6 = 32
print(f"\ndot product v1 . v2 = {dot}")

# Batched matmul
batch_A = torch.rand(4, 3, 2)            # batch of 4 matrices [3x2]
batch_B = torch.rand(4, 2, 5)            # batch of 4 matrices [2x5]
batch_result = torch.bmm(batch_A, batch_B)
print(f"\nbmm: {batch_A.shape} @ {batch_B.shape} -> {batch_result.shape}")

# Transpose trick for same-shape matmul
W = torch.rand(3, 4)
X = torch.rand(3, 4)
result = torch.matmul(W, X.T)            # [3,4] @ [4,3] -> [3,3]
print(f"\nW @ X.T : {W.shape} @ {X.T.shape} -> {result.shape}")

print("\nDone.")
