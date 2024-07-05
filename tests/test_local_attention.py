import pytest

import torch
from local_attention.transformer import LocalMHA, DynamicPositionBias

@pytest.mark.parametrize('use_rotary_pos_emb', (True, False))
@pytest.mark.parametrize('use_attn_bias', (True, False))
@pytest.mark.parametrize('exact_windowsize', (True, False))
@pytest.mark.parametrize('window_size', (128, 1000))
def test_cache(
    use_rotary_pos_emb,
    use_attn_bias,
    exact_windowsize,
    window_size
):

    attn_bias = None
    if use_attn_bias:
        attn_bias_mlp = DynamicPositionBias(
            dim = 32,
            heads = 8
        )

        attn_bias_mlp.eval()
        attn_bias = attn_bias_mlp(window_size, window_size * 2)

    mha = LocalMHA(
      dim = 512,
      heads = 8,
      gate_values_per_head = False,
      window_size = window_size,
      exact_windowsize = exact_windowsize,
      use_rotary_pos_emb = use_rotary_pos_emb,
      causal = True
    )

    mha.eval()

    seq = torch.randn(2, 1000, 512)

    _, cache = mha(seq, return_cache = True)

    next_token = torch.randn(2, 1, 512)
    seq_with_next_token = torch.cat((seq, next_token), dim = 1)

    out_without_cache = mha(seq_with_next_token, attn_bias = attn_bias)[:, -1]
    out_with_cache = mha(next_token, cache = cache, attn_bias = attn_bias)[:, -1]

    assert torch.allclose(out_without_cache, out_with_cache, atol = 1e-6)
