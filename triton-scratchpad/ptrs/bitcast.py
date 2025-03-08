import triton
import triton.language as tl

import torch

@triton.jit
def ptr_cast(input, output1, output2):
    tl.static_assert(input.dtype == tl.pointer_type(tl.int8))
    # tl.static_assert(output1.dtype == tl.pointer_type(tl.uint8))
    orig_ptr = input + 1
    ptr_1 = orig_ptr
    v1 = tl.load(ptr_1)
    ptr_16_bit = orig_ptr.to(tl.pointer_type(tl.int32))
    ptr_2 = ptr_16_bit
    v2 = tl.load(ptr_2)
    tl.device_print("test", v2)
    tl.store(output1, v1)
    tl.store(output1 + 1, v2)

    tl.store(output2, v1)
    tl.store(output2 + 1, v2)


"""
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
       device='cuda:0', dtype=torch.int8)
tensor([       0, 50462976,       -1,       -1,       -1,       -1,       -1,
              -1,       -1,       -1,       -1,       -1,       -1,       -1,
              -1,       -1], device='cuda:0', dtype=torch.int32)
tensor([ 0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       device='cuda:0', dtype=torch.int8)
---

>>> (3 << 24) | (2 << 16) | (1 << 8) | 0
50462976
"""


# def main():
#     buffer = torch.arange(0, 16, dtype=torch.int8, device='cuda')
#     print(buffer.cpu().numpy().tobytes())
#     output1 = torch.full((16,), -1, dtype=torch.int32, device='cuda')
#     output2 = torch.full((16,), -1, dtype=torch.int8, device='cuda')
#     print(buffer)
#     # print(output)
#     ptr_cast[(1,)](buffer, output1, output2)
#     # print(buffer)
#     print(output1)
#     print(output2)

# main()


def test_cast():
    @triton.jit
    def no_cast(in_ptr0, out_ptr0):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        cond = tl.load(in_ptr0)
        offsets = tl.arange(0, 16)
        ptr = (in_ptr0 + tl.load(in_ptr0 + offsets))
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        if (cond):
            ptr += 2 * tl.load(offsets+ ptr)
        else:
            ptr += 2 * tl.load(offsets+ ptr) + 10
        # cast again
        ptr = (ptr) + 4 * tl.load(offsets+ ptr)
        # We should be loading 7
        ptr_i32 = ptr + 5 * tl.load(offsets+ ptr)
        a = tl.load(ptr_i32 + offsets)
        tl.store(out_ptr0 + offsets, a)


    @triton.jit
    def no_cast_one_branch_with_1_base_else_multiples(in_ptr0, out_ptr0):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        cond = tl.load(in_ptr0)
        offsets = tl.arange(0, 16)
        ptr = (in_ptr0 + tl.load(in_ptr0 + offsets))
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        if (cond):
            ptr += 2 * tl.load(offsets+ ptr)
        else:
            ptr_0 = in_ptr0 + tl.arange(0, 8)
            ptr_1 = out_ptr0 + tl.arange(0, 8)
            ptr = tl.cat(ptr_0, ptr_1, can_reorder=True)

        # cast again
        ptr = (ptr) + 4 * tl.load(offsets+ ptr)
        # We should be loading 7
        ptr_i32 = ptr + 5 * tl.load(offsets+ ptr)
        a = tl.load(ptr_i32 + offsets)
        tl.store(out_ptr0 + offsets, a)


    @triton.jit
    def cast(in_ptr0, out_ptr0, idx):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        ptr = (in_ptr0 + 1 + idx * 2).to(tl.pointer_type(tl.int64))
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        ptr += 2
        ptr += idx
        ptr += 3
        # cast again
        ptr = (ptr).to(tl.pointer_type(tl.int16)) + 4
        ptr += idx
        ptr += 3
        # We should be loading 7
        ptr_i32 = ptr.to(tl.pointer_type(tl.int32))
        a = tl.load(ptr_i32)
        tl.store(out_ptr0, a)

    @triton.jit
    def cast_with_int_ptr(in_ptr0, out_ptr0, idx):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        ptr_0 = (in_ptr0 + 111).to(tl.pointer_type(tl.int8)) + 10
        ptr_0 = ptr_0.to(tl.pointer_type(tl.int32))

        t0 = out_ptr0.to(tl.int64)
        out_ptr0 += t0
        out_ptr0 += 9

        t = out_ptr0.to(tl.int64) % 10 # ptr_to_int
        ptr = (ptr_0 + 1 + t).to(tl.pointer_type(tl.int64))
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        ptr += 2
        ptr += idx
        ptr += 3
        # cast again
        ptr = (ptr).to(tl.pointer_type(tl.int16)) + 4
        ptr += idx
        ptr += 3
        # We should be loading 7
        ptr_i32 = ptr.to(tl.pointer_type(tl.int32))
        a = tl.load(ptr_i32)
        t = (t + idx).to(tl.pointer_type(tl.int32))
        tl.store(t, a)


    @triton.jit
    def cast_tensor_ptr(in_ptr0, out_ptr0):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        in_ptr = (in_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        in_ptr += 2
        # cast again
        in_ptr = (in_ptr).to(tl.pointer_type(tl.int16)) + 4
        # We should be loading 7
        in_ptr_i32 = in_ptr.to(tl.pointer_type(tl.int32))
        a = tl.load(in_ptr_i32)

        out_ptr = (out_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
        out_ptr += 2
        out_ptr_i32 = out_ptr.to(tl.pointer_type(tl.int32))
        tl.store(out_ptr_i32, a)

    @triton.jit
    def bitcast_ptr_as_src(in_ptr0, out_ptr0):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        in_ptr = (in_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
        # now each increment in ptr is equivalent to 2 increments in the original ptr
        #    ptr         ptr    ptr
        #     v           v     v
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #  ^     ^
        in_ptr += 2
        # We should be loading 7
        a = tl.load(in_ptr)

        out_ptr = (out_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
        out_ptr += 2
        tl.store(out_ptr, a)


    # buffer = torch.arange(0, 16, dtype=torch.int32, device='cuda')
    # output1 = torch.full((16,), -1, dtype=torch.int32, device='cuda')
    # cast[(1,)](buffer, output1)
    # # print(buffer)
    # print(output1)


    src = triton.compiler.ASTSource(
        fn=bitcast_ptr_as_src,
        signature="*i32,*i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])


# test_control_flow()
# test_mixed_structured_and_unstructured()
# test_intermediate_ptr_as_base()
test_cast()
