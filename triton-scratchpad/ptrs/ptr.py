import triton
import triton.language as tl

import torch



def test_mixed_structured_and_unstructured():
    @triton.jit
    def mixed(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        pid = tl.program_id(0)
        base_ptr = in_ptr0
        base_ptr += 10

        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs_0 = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1


        if pid == 0:
            base_ptr += 20
        else:
            base_ptr = in_ptr1

        load_ptrs_1 = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        store_ptrs = out_ptr0 + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        vals_0 = tl.load(load_ptrs_0)
        vals_1 = tl.load(load_ptrs_1)

        tl.store(store_ptrs, vals_0)
        tl.store(store_ptrs, vals_1)

    src = triton.compiler.ASTSource(
        fn=mixed,
        signature="*fp32,*fp32,i32,i32,*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])


def test_mixed_structured_and_unstructured():
    @triton.jit
    def mixed(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        pid = tl.program_id(0)
        base_ptr = in_ptr0
        base_ptr += 10

        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs_0 = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1


        if pid == 0:
            base_ptr += 20
        else:
            base_ptr = in_ptr1

        load_ptrs_1 = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        store_ptrs = out_ptr0 + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        vals_0 = tl.load(load_ptrs_0)
        vals_1 = tl.load(load_ptrs_1)

        tl.store(store_ptrs, vals_0)
        tl.store(store_ptrs, vals_1)

    src = triton.compiler.ASTSource(
        fn=mixed,
        signature="*fp32,*fp32,i32,i32,*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

# Test if triton-to-structured works with an intermediate pointer as base
def test_intermediate_ptr_as_base():
    # triton-to-structured does not run on this code because of the select op
    @triton.jit
    def control_flow(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        pid = tl.program_id(0)
        if pid == 0:
            base_ptr = in_ptr0
            out_base_ptr = out_ptr0
        else:
            base_ptr = in_ptr1
            out_base_ptr = out_ptr1

        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1
        vals = tl.load(load_ptrs)

        store_ptrs = out_base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        tl.store(store_ptrs, vals)

    @triton.jit
    def intermediate_as_base(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        base_ptr = in_ptr0
        out_base_ptr = out_ptr0

        base_ptr += stride_0
        out_base_ptr += stride_1

        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1
        vals = tl.load(load_ptrs)

        store_ptrs = out_base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        tl.store(store_ptrs, vals)


    # This crashes. The crash can be fixed using https://github.com/microsoft/triton-shared/pull/215/files
    # But we still cannot use the return value from the loop as intermediate.
    @triton.jit
    def intermediate_as_base_returned_from_loop(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        base_ptr = in_ptr0
        out_base_ptr = out_ptr0

        for i in range(0, 10):
            base_ptr += stride_0
            out_base_ptr += stride_1

        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1
        vals = tl.load(load_ptrs)

        store_ptrs = out_base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        tl.store(store_ptrs, vals)


    # This still doesn't work. Similar to `control_flow`
    @triton.jit
    def intermediate_as_base_conditional_offset(in_ptr0, in_ptr1, stride_0, stride_1, out_ptr0, out_ptr1):
        pid = tl.program_id(0)
        base_ptr = in_ptr0
        out_base_ptr = out_ptr0

        if pid == 0:
            base_ptr += 10


        offs_am = tl.arange(0, 16)
        offs_k = tl.arange(0, 32)
        load_ptrs = base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1
        vals = tl.load(load_ptrs)

        store_ptrs = out_base_ptr + offs_am[:, None] * stride_0 + offs_k[None, :] * stride_1

        tl.store(store_ptrs, vals)

    src = triton.compiler.ASTSource(
        fn=intermediate_as_base_conditional_offset,
        signature="*fp32,*fp32,i32,i32,*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])



def test_bug():
    @triton.jit
    def simple_cf_into_structured_load(in_ptr0, in_ptr1, out_ptr, idx):
        if idx == 1:
            in_ptr += idx * 2
        else:
            in_ptr += idx

        in_ptr += 6

        offsets = tl.arange(0, 4)
        ptrs = in_ptr + offsets
        vals = tl.load(ptrs)
        tl.store(out_ptr + offsets, vals)

    src = triton.compiler.ASTSource(
        fn=simple_cf_into_structured_load,
        signature="*fp32,*fp32,*fp32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

# test_control_flow()
# test_mixed_structured_and_unstructured()
# test_intermediate_ptr_as_base()
test_bug()