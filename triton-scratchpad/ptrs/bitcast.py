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

@triton.jit
def nested_use_same_level_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i1 in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j1 in range(0, 2):
            a_ptrs += 2 * stride_n

        for i6 in range(0, 2):
            a1 = tl.load(a_ptrs)
            a_ptrs += 2 * stride_n
            a3 = tl.load(a_ptrs)
            tl.store(c_ptrs, a1)
            c_ptrs += 2 * stride_n

            c_ptrs += 2 * stride_n
            tl.store(c_ptrs, a3)
            c_ptrs += 2 * stride_n
            a_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n

def test_nested2_use_same_level_loop_result():
    n_rows = 4
    n_cols = 32
    grid = lambda meta: (n_cols // 4,)
    expected = torch.tensor([[ 4,  5,  0,  0,  6,  7,  8,  9,  0,  0, 10, 11, 18, 19,  0,  0, 20, 21,
         22, 23,  0,  0, 24, 25,  0,  0,  0,  0,  0,  0,  0,  0],
        [36, 37,  0,  0, 38, 39, 40, 41,  0,  0, 42, 43, 50, 51,  0,  0, 52, 53,
         54, 55,  0,  0, 56, 57,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
       device='cpu', dtype=torch.int32)



    src = triton.compiler.ASTSource(
        fn=nested_use_same_level_loop_results,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')



def test_cast():
    @triton.jit
    def tensor_ptr(in_ptr0, out_ptr0):
        ints = tl.load(in_ptr0 + tl.arange(0, 16)).to(tl.int64)
        ptrs = ints.to(tl.pointer_type(tl.int32))
        masks = tl.load(ptrs + 16).to(tl.int1)
        vals = tl.load(ptrs, mask=masks)
        out_ptrs = out_ptr0 + tl.arange(0, 16)
        ints_2 = out_ptrs.to(tl.int64) + vals
        out_ptrs = out_ptr0 + tl.arange(0, 16)
        out_ptrs_i64 = out_ptrs.to(tl.pointer_type(tl.int64))
        out_ptrs_i64 += 2
        out_ptrs_i32 = out_ptrs_i64.to(tl.pointer_type(tl.int32))
        tl.store(out_ptrs_i32, ints_2.to(tl.int32), mask=masks)


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
    def ptr_cat(in_ptr0, out_ptr0, mask_ptr):
        offsets = tl.arange(0, 16)
        ptr_0 = in_ptr0 + tl.arange(0, 8)
        ptr_1 = out_ptr0 + tl.arange(0, 8)
        ptr = tl.cat(ptr_0, ptr_1, can_reorder=True)
        ptr_true = ptr + 4 * tl.load(offsets + ptr)
        ptr_false = ptr + 5 * tl.load(offsets + ptr)
        masks = tl.load(mask_ptr + offsets)
        ptr_load = tl.where(masks, ptr_true, ptr_false)
        a = tl.load(ptr_load + offsets, mask=masks)
        tl.store(out_ptr0 + offsets, a, mask=masks)


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
        ptr = (in_ptr0 + idx).to(tl.pointer_type(tl.int64))
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
        tl.store(out_ptr0, a)


    @triton.jit
    def cast_tensor_ptr(in_ptr0, out_ptr0, idx):
        tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
        ptr = (in_ptr0 + idx + tl.arange(0, 4)).to(tl.pointer_type(tl.int64))
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
        tl.store(out_ptr0 + tl.arange(0, 4), a)

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


    # @triton.jit
    # def cast_tensor_ptr(in_ptr0, out_ptr0):
    #     tl.static_assert(in_ptr0.dtype == tl.pointer_type(tl.int32))
    #     in_ptr = (in_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
    #     # now each increment in ptr is equivalent to 2 increments in the original ptr
    #     #    ptr         ptr    ptr
    #     #     v           v     v
    #     # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #     #  ^     ^
    #     in_ptr += 2
    #     # cast again
    #     in_ptr = (in_ptr).to(tl.pointer_type(tl.int16)) + 4
    #     # We should be loading 7
    #     in_ptr_i32 = in_ptr.to(tl.pointer_type(tl.int32))
    #     a = tl.load(in_ptr_i32)

    #     out_ptr = (out_ptr0 + 1).to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
    #     out_ptr += 2
    #     out_ptr_i32 = out_ptr.to(tl.pointer_type(tl.int32))
    #     tl.store(out_ptr_i32, a)

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


    @triton.jit
    def int_to_ptr(out_ptr, some_address):
        tl.static_assert(some_address.dtype == tl.int64)
        ptrs = some_address.to(tl.pointer_type(tl.int64)) + tl.arange(0, 16)
        vals = tl.load(ptrs)
        tl.store(out_ptr + tl.arange(0, 16), vals)


    @triton.jit
    def ptr_as_values(in_ptr_0, in_ptr_1, offset_ptr_0, offset_ptr_1, mask_ptr, out_ptr):
        tl.static_assert(mask_ptr.dtype == tl.pointer_type(tl.int1))
        masks = tl.load(mask_ptr + tl.arange(0, 16))
        ptrs_0 = in_ptr_0 + tl.load(offset_ptr_0 + tl.arange(0, 16))
        ptrs_1 = in_ptr_1 + tl.load(offset_ptr_1 + tl.arange(0, 16))
        ptrs = tl.where(masks, ptrs_0, ptrs_1)
        values = tl.load(ptrs)
        tl.store(out_ptr + tl.arange(0, 16), values)


    @triton.jit
    def bitcast(in_ptr, out_ptr):
        tl.static_assert(in_ptr.dtype == tl.pointer_type(tl.int64))
        tl.static_assert(out_ptr.dtype == tl.pointer_type(tl.int32))
        in_ptr += 8
        in_ptr_i32 = in_ptr.to(tl.pointer_type(tl.int32))
        in_ptr_i32 += 12
        values = tl.load(in_ptr_i32 + tl.arange(0, 16) // 8)
        tl.store(out_ptr + tl.arange(0, 16), values)



    buffer = torch.arange(0, 32, dtype=torch.int32, device='cuda')
    output1 = torch.full((32,), -1, dtype=torch.int32, device='cuda')
    cast_tensor_ptr[(1,)](buffer, output1, 1)
    # print(buffer)
    print(output1)

    src = triton.compiler.ASTSource(
        fn=ptr_cat,
        signature="*i32,*i32,*i1",
    )


    # src = triton.compiler.ASTSource(
    #     fn=bitcast_ptr_as_src,
    #     signature="*i32,*i32",
    # )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])


# test_control_flow()
# test_mixed_structured_and_unstructured()
# test_intermediate_ptr_as_base()
test_cast()
# test_nested2_use_same_level_loop_result()

