import torch

import triton
import triton.language as tl
import torch


@triton.jit
def try_complex_lut(in_ptr0, out_ptr0, idx):
    a = tl.load(in_ptr0)
    b = a.to(tl.pointer_type(tl.pointer_type(tl.int64)))
    tl.store(out_ptr0,)


@triton.jit
def try_complex_lut(in_ptr0, out_ptr0, idx):
    tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
    tl.static_assert(out_ptr0.dtype == tl.pointer_type(tl.int32))
    # tl.static_assert(idx.dtype == tl.int32)
    offsets = tl.arange(0, 4)
    ptr = (in_ptr0 + idx).to(tl.pointer_type(tl.int64)) + offsets
    # We're at 1 now, assume idx is 1
    # now each increment in ptr is equivalent to 2 increments in the original ptr
    #    ptr         ptr    ptr
    #     v           v     v
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #  ^     ^
    ptr += 2
    ptr += idx
    ptr += 3
    # We should be at 1 + (2 + 1 + 3) * 2 = 13
    # cast again
    ptr = (ptr).to(tl.pointer_type(tl.int16)) + 4 # at 15
    ptr += idx
    ptr += 3
    # at 17
    # We should be loading 17
    ptr_i32 = ptr.to(tl.pointer_type(tl.int32))
    a = tl.load(ptr_i32)
    tl.store(out_ptr0 + offsets, a)

def main():
    input = torch.arange(0, 64, dtype=torch.int32, device='cuda')
    output1 = torch.full((64,), -1, dtype=torch.int32, device='cuda')
    # ptr_tensor_0 = torch.full((32,), 11, dtype=torch.int64, device='cpu')
    # ptr_tensor_1 = torch.full((32,), 11, dtype=torch.int32, device='cpu')

    try_complex_lut[(1,)](input, output1, 1)

    print('hehehaha000000')
    print(output1)

main()
