"""
PyTorch Datatypes & Most Common Errors
=======================================
Covers: common dtypes, and the 3 most common tensor errors:
        - dtype mismatch
        - shape mismatch
        - device mismatch
"""

import torch

# =============================================================================
# 1. DATATYPES
# =============================================================================

print("=" * 50)
print("DATATYPES")
print("=" * 50)

# Default float (float32) - most common for ML
f32 = torch.tensor([1.0, 2.0, 3.0])                          # default: float32
print(f"float32 (default) : {f32.dtype}")

# Explicit dtypes
f16  = torch.tensor([1.0, 2.0], dtype=torch.float16)         # half precision
f64  = torch.tensor([1.0, 2.0], dtype=torch.float64)         # double precision
i8   = torch.tensor([1, 2],     dtype=torch.int8)            # 8-bit int
i32  = torch.tensor([1, 2],     dtype=torch.int32)           # 32-bit int
i64  = torch.tensor([1, 2],     dtype=torch.int64)           # long (default int)
boo  = torch.tensor([True, False], dtype=torch.bool)         # boolean

print(f"float16           : {f16.dtype}")
print(f"float64           : {f64.dtype}")
print(f"int8              : {i8.dtype}")
print(f"int32             : {i32.dtype}")
print(f"int64             : {i64.dtype}")
print(f"bool              : {boo.dtype}")

# Check & cast dtype
t = torch.tensor([1.0, 2.0, 3.0])
print(f"\nOriginal dtype    : {t.dtype}")
t_int = t.type(torch.int32)                                   # cast to int32
print(f"After .type(int32): {t_int.dtype} | values: {t_int}")

# =============================================================================
# 2. COMMON ERROR 1 - DTYPE MISMATCH
# =============================================================================

print("\n" + "=" * 50)
print("ERROR 1: dtype mismatch")
print("=" * 50)

a = torch.tensor([1.0, 2.0], dtype=torch.float32)
b = torch.tensor([1,   2  ], dtype=torch.int32)

print(f"a dtype: {a.dtype}  |  b dtype: {b.dtype}")

# Operations between incompatible dtypes can raise RuntimeError
try:
    result = a + b
    print(f"Result (auto-promoted in newer PyTorch): {result}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")

# FIX: cast b to match a
b_fixed = b.type(torch.float32)
result = a + b_fixed
print(f"Fixed -> cast b to float32: {result}")

# =============================================================================
# 3. COMMON ERROR 2 - SHAPE MISMATCH
# =============================================================================

print("\n" + "=" * 50)
print("ERROR 2: shape mismatch")
print("=" * 50)

x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])          # shape [2, 3]
y = torch.tensor([[1, 2],
                   [3, 4]])             # shape [2, 2]

print(f"x shape: {x.shape}  |  y shape: {y.shape}")

# Adding incompatible shapes raises RuntimeError
try:
    result = x + y
except RuntimeError as e:
    print(f"RuntimeError: {e}")

# FIX: use matching shapes
z = torch.tensor([[7, 8, 9],
                   [10, 11, 12]])       # shape [2, 3] - matches x
result = x + z
print(f"Correct add (same shape): {result}")

# Matmul shape rule: (m, n) @ (n, p) -> (m, p)
a_mat = torch.rand(2, 3)
b_mat = torch.rand(3, 4)
print(f"\nmatmul: {a_mat.shape} @ {b_mat.shape} -> {torch.matmul(a_mat, b_mat).shape}")

try:
    bad = torch.matmul(a_mat, a_mat)    # [2,3] @ [2,3] -> inner dims don't match
except RuntimeError as e:
    print(f"matmul mismatch -> RuntimeError: {e}")

# FIX: transpose one matrix
result = torch.matmul(a_mat, a_mat.T)  # [2,3] @ [3,2] -> [2,2]
print(f"Fixed with .T -> shape: {result.shape}")

# =============================================================================
# 4. COMMON ERROR 3 - DEVICE MISMATCH
# =============================================================================

print("\n" + "=" * 50)
print("ERROR 3: device mismatch")
print("=" * 50)

cpu_tensor = torch.tensor([1.0, 2.0, 3.0])                   # lives on CPU
print(f"cpu_tensor device: {cpu_tensor.device}")

if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to("cuda")                        # move to GPU
    print(f"gpu_tensor device: {gpu_tensor.device}")

    # Adding CPU and GPU tensors raises RuntimeError
    try:
        result = cpu_tensor + gpu_tensor
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

    # FIX: move both to the same device
    cpu_on_gpu = cpu_tensor.to("cuda")
    result = cpu_on_gpu + gpu_tensor
    print(f"Fixed -> both on GPU: {result.device}")
else:
    print("CUDA not available - running on CPU only.")
    print("In practice, device mismatch looks like:")
    print("  RuntimeError: Expected all tensors to be on the same device,")
    print("  but found at least two devices, cuda:0 and cpu!")
    print("\nFIX: use tensor.to(device) to move tensors to the same device.")
    print("  device = 'cuda' if torch.cuda.is_available() else 'cpu'")
    print("  tensor = tensor.to(device)")

print("\nDone.")