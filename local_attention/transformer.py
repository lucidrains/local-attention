from __future__ import annotations
from copy import deepcopy
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, einsum

from local_attention.local_attention import LocalAttention
from local_attention.rotary import apply_rotary_pos_emb

from hyper_connections import get_init_and_expand_reduce_stream_functions

# helper function

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling functions

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# multi-head attention

class LocalMHA(Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.causal = causal
        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = self.exact_windowsize,
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        attn_bias = None,
        cache = None,
        return_cache = False
    ):
        seq_len = x.shape[-2]

        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        if exists(cache):
            assert seq_len == 1

            assert self.causal and not exists(mask), 'only allow caching for specific configuration'

            ck, cv = cache

            q = q * (q.shape[-1] ** -0.5)

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            effective_window_size = self.attn_fn.look_backward * self.window_size

            if self.exact_windowsize:
                kv_start_index = -(effective_window_size + 1)
            else:
                seq_len = k.shape[-2]
                kv_start_index = -(effective_window_size + (seq_len % self.window_size))

            k, v = tuple(t[..., kv_start_index:, :] for t in (k, v))

            if exists(self.attn_fn.rel_pos):
                rel_pos = self.attn_fn.rel_pos
                pos_emb, xpos_scale = rel_pos(k)
                q, k = apply_rotary_pos_emb(q, k, pos_emb, scale = xpos_scale)

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(attn_bias):
                k_len = k.shape[-2]
                attn_bias = attn_bias[..., -1:, -k_len:]
                assert attn_bias.shape[-1] == sim.shape[-1]
                sim = sim + attn_bias

            attn = sim.softmax(dim = -1)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        else:
            out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if return_cache:
            kv = torch.stack((k, v))

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, kv

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# dynamic positional bias

class DynamicPositionBias(Module):
    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads)
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, i, j):
        device = self.device
        assert j >= i

        rel_dist = torch.arange(j, dtype = torch.float, device = device)
        bias = self.mlp(rearrange(rel_dist, '... -> ... 1'))

        i_seq = torch.arange(j - i, j, device = device)
        j_seq = torch.arange(j, device = device)

        rel_dist_indices = (rearrange(i_seq, 'i -> i 1') - rearrange(j_seq, 'j -> 1 j')).abs()

        bias = rearrange(bias[rel_dist_indices], 'i j h -> h i j')
        return bias

# main transformer class

Cache = namedtuple('Cache', ['cache_kv', 'maybe_cached_attn_bias'])

class LocalTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal = True,
        local_attn_window_size = 512,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ignore_index = -1,
        use_xpos = False,
        xpos_scale_base = None,
        use_dynamic_pos_bias = False,
        global_attn_layer: Module | None = None,
        layers_insert_global_attn: tuple[int, ...] | None = None,
        num_residual_streams = 4,
        **kwargs
    ):
        super().__init__()

        self.has_embed_unembed = exists(num_tokens)

        if self.has_embed_unembed:
            self.token_emb = nn.Embedding(num_tokens, dim)
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # allow for inserting global attention or memory layers

        layers_insert_global_attn = default(layers_insert_global_attn, tuple(range(1, depth + 1)))
        assert all([0 < layer <= depth for layer in layers_insert_global_attn])

        global_attn_layers = set(layers_insert_global_attn)

        self.global_layers = ModuleList([])

        # define modules throughout layers

        for index in range(depth):
            layer = index + 1

            self.global_layers.append(init_hyper_conn(dim = dim, branch = deepcopy(global_attn_layer)) if exists(global_attn_layer) and layer in global_attn_layers else None)

            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim = dim, branch = LocalMHA(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal, window_size = local_attn_window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base, use_rotary_pos_emb = not use_dynamic_pos_bias, prenorm = True, **kwargs)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.ignore_index = ignore_index

        if self.has_embed_unembed:
            self.to_logits = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        seq_len,
        temperature = 1.,
        filter_thres = 0.9,
        use_kv_cache = True,
        **kwargs
    ):
        assert self.has_embed_unembed
        assert temperature >= 0.

        n, device = prime.shape[1], prime.device

        out = prime

        cache = None

        for _ in range(seq_len):

            logits, new_cache = self.forward(
                out[:, -self.max_seq_len:],
                cache = cache,
                return_cache = True,
                **kwargs
            )

            if use_kv_cache:
                cache = new_cache

            filtered_logits = top_k(logits[:, -1], thres = filter_thres)

            if temperature == 0.:
                sampled = filtered_logits.argmax(dim = -1, keepdim = True)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sampled = torch.multinomial(probs, 1)

            out = torch.cat((out, sampled), dim = -1)

        return out[:, n:]

    def forward(
        self,
        x,
        mask = None,
        cache = None,
        return_loss = False,
        return_cache = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n, device = x.shape[1], x.device

        if self.has_embed_unembed:
            x = self.token_emb(x)

            assert n <= self.max_seq_len
            x = x + self.pos_emb(torch.arange(n, device = device))

        # handle old and new cache

        has_cache = exists(cache)
        cached_kv = cached_attn_bias = None

        if has_cache:
            cached_kv, cached_attn_bias = cache

        new_cached_kv = []
        iter_cached_kv = iter(default(cached_kv, []))

        if has_cache:
            x = x[:, -1:]

        # dynamic pos bias

        attn_bias = cached_attn_bias

        if not exists(attn_bias) and exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        x = self.expand_streams(x)

        for (attn, ff), global_layer in zip(self.layers, self.global_layers):

            if exists(global_layer):
                x = global_layer(x)

            x, layer_cached_kv = attn(
                x,
                mask = mask,
                attn_bias = attn_bias,
                return_cache = True,
                cache = next(iter_cached_kv, None)
            )

            new_cached_kv.append(layer_cached_kv)

            x = ff(x)

        x = self.reduce_streams(x)

        if not self.has_embed_unembed:
            return x

        # to logits

        logits = self.to_logits(x)

        if not return_loss:

            if not return_cache:
                return logits

            return logits, Cache(new_cached_kv, attn_bias)

        # cross entropy loss

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
