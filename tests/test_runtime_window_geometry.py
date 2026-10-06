import pytest
import torch
import torch.nn.functional as F

from local_attention import LocalAttention
from local_attention.transformer import LocalMHA


def rotate_globally(t, relative, inverse=False):
    n = t.shape[-2]
    positions = torch.arange(n, dtype=t.dtype, device=t.device)
    angle = positions[:, None] * relative.inv_freq
    angle = torch.cat((angle, angle), dim=-1)
    first, second = t.chunk(2, dim=-1)
    rotated = t * angle.cos() + torch.cat((-second, first), dim=-1) * angle.sin()
    if relative.use_xpos:
        scales = relative.scale ** (positions[:, None] / relative.scale_base)
        scales = torch.cat((scales, scales), dim=-1)
        rotated = rotated * (scales.reciprocal() if inverse else scales)
    return rotated


def dense_local_formula(module, q, k, v, window_size=None, mask=None):
    window_size = module.window_size if window_size is None else window_size
    n = q.shape[-2]
    positions = torch.arange(n)
    query = positions[:, None]
    key = positions[None, :]
    allowed = ((query // window_size - key // window_size) <= module.look_backward)
    allowed = allowed & ((key // window_size - query // window_size) <= module.look_forward)
    if module.causal:
        allowed = allowed & (key <= query)
    if module.exact_windowsize:
        allowed = allowed & (query - key <= window_size * module.look_backward)
        allowed = allowed & (key - query <= window_size * module.look_forward)
    if mask is not None:
        allowed = allowed[None, None] & mask[:, None, None, :]
    if module.rel_pos is not None:
        q = rotate_globally(q, module.rel_pos)
        k = rotate_globally(k, module.rel_pos, inverse=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                         scale=module.scale)


def dense_mha_formula(module, x):
    if module.norm is not None:
        x = F.layer_norm(x, module.norm.normalized_shape, module.norm.weight,
                         module.norm.bias, module.norm.eps)
    batch, n, _ = x.shape
    projected = F.linear(x, module.to_qkv.weight)
    q, k, v = projected.chunk(3, dim=-1)
    q, k, v = [t.reshape(batch, n, module.heads, -1).transpose(1, 2) for t in (q, k, v)]
    out = dense_local_formula(module.attn_fn, q, k, v)
    return F.linear(out.transpose(1, 2).reshape(batch, n, -1), module.to_out.weight)


@pytest.mark.parametrize('seq_len', [7, 12, 13])
@pytest.mark.parametrize('look_backward', [1, 2])
def test_runtime_xpos_window_matches_dense_receptive_field(seq_len, look_backward):
    torch.manual_seed(47)
    module = LocalAttention(window_size=4, causal=True, exact_windowsize=True,
                            look_backward=look_backward, dim=8, use_xpos=True,
                            xpos_scale_base=64, autopad=True).double()
    q, k, v = [torch.randn(2, 2, seq_len, 8, dtype=torch.float64, requires_grad=True)
               for _ in range(3)]
    mask = torch.ones(2, seq_len, dtype=torch.bool)
    actual = module(q, k, v, mask=mask, window_size=6)
    expected = dense_local_formula(module, q, k, v, window_size=6, mask=mask)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
