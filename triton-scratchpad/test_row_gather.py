import torch

"""
Q_PER_K 2
BLOCK_DMODEL 128
BLOCK_M 64

Q_PER_K 2
BLOCK_DMODEL 128
BLOCK_M 16

stride_qs 384
stride_qh 192
stride_qg 96
"""

stride_qs = 384
stride_qh = 192
stride_qg = 96

BLOCK_M = 16
BLOCK_DMODEL = 128
Q_PER_K = 4

offs_m = torch.arange(0, BLOCK_M)
offs_d = torch.arange(0, BLOCK_DMODEL)

offs_g_q = offs_m[:, None] % Q_PER_K + 5
print(offs_g_q)
offs_s_q = offs_m[:, None] // Q_PER_K + 7
print(offs_s_q)

ptr = offs_s_q * stride_qs + offs_g_q * stride_qg + offs_d[None, :]
# torch.set_printoptions(threshold=10_000)
print('p')
print(ptr)
print(ptr.shape)

stride_ms = 128
stride_mg = 32

offset_mz = (
    offs_s_q * stride_ms + offs_g_q * stride_mg
)

print('offset_mz')
print(offset_mz)
print(offset_mz.shape)


stride_os = 12288
stride_og = 3072
O_ptr =  offs_s_q * stride_os + offs_g_q * stride_og + offs_d[None, :]
print('O_ptr')
print(O_ptr)
print(O_ptr.shape)

# next question: combination of Q_Per_k and BLOCK_M?
# is BLOCK_M always >= Q_PER_K?
# yeah block_m = max(16, prompt_q_per_k * max_attn_chunk_size)
#
# min 16
# PROMPT_BLOCK_M = max(16, PROMPT_Q_PER_K * s_q_max)
# s_q_max = max_attention_chunk_size we set in the test
# ok it doesn't matter because the row starting index is always
# now i need to verify how many loads these affect
