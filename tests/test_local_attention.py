import pytest

import torch
from local_attention.transformer import LocalMHA

@pytest.mark.parametrize('use_rotary_pos_emb', (True, False))
def test_cache(
        use_rotary_pos_emb
    ):

    mha = LocalMHA(
      dim = 512,
      gate_values_per_head = False,
      window_size = 1000,
      exact_windowsize = True,
      use_rotary_pos_emb = use_rotary_pos_emb,
      causal = True
    )

    mha.eval()

    seq = torch.randn(2, 1000, 512)

    _, cache = mha(seq, return_cache = True)

    next_token = torch.randn(2, 1, 512)
    seq_with_next_token = torch.cat((seq, next_token), dim = 1)

    out_without_cache = mha(seq_with_next_token)[:, -1]
    out_with_cache = mha(next_token, cache = cache)[:, -1]

    assert torch.allclose(out_without_cache, out_with_cache, atol = 1e-7)
