"""
model_patches.py -- Runtime monkey-patches for WanModel block causal training.

This module replaces forward methods at runtime to add block_causal + chunk_size_tokens support.

Usage:
    from model_patches import patch_model_for_block_causal

    model = WanModel.from_pretrained(...)
    patch_model_for_block_causal(model)

    # Now model.forward() accepts block_causal=True, chunk_size_tokens=N
    model(x=..., t=..., ..., block_causal=True, chunk_size_tokens=1560)

When block_causal=False (default), behavior is identical to the original
unpatched model, so inference via WanI2VCausal still works without patching.
"""

import types

import torch


# =============================================================================
# Patched forward: WanSelfAttention
# =============================================================================


def block_causal_self_attn_forward(
    self, x, seq_lens, grid_sizes, freqs, frame_offset=0,
    use_cache=False, block_causal=False, chunk_size_tokens=0,
):
    """WanSelfAttention.forward with block causal support.

    When block_causal=False, falls through to original KV-cache / vanilla path.
    When block_causal=True, processes chunks sequentially: each chunk attends
    bidirectionally within itself and read-only to detached past chunk KV.
    """
    from wan.modules.attention import flash_attention
    from wan.modules.model import rope_apply

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    q = rope_apply(q, grid_sizes, freqs, frame_offset=frame_offset)
    k = rope_apply(k, grid_sizes, freqs, frame_offset=frame_offset)

    if block_causal and chunk_size_tokens > 0:
        # ----- Block causal: chunk-sequential with accumulated past KV -----
        total_tokens = int(seq_lens[0].item()) if seq_lens[0].dim() == 0 else int(seq_lens[0])
        num_chunks = (total_tokens + chunk_size_tokens - 1) // chunk_size_tokens

        past_k_list = []
        past_v_list = []
        output_chunks = []

        for c in range(num_chunks):
            start = c * chunk_size_tokens
            end = min(start + chunk_size_tokens, total_tokens)
            chunk_len = end - start

            q_c = q[:, start:end]
            k_c = k[:, start:end]
            v_c = v[:, start:end]

            if past_k_list:
                # Concatenate detached past KV with current chunk
                past_k = torch.cat(past_k_list, dim=1).detach()
                past_v = torch.cat(past_v_list, dim=1).detach()
                k_full = torch.cat([past_k, k_c], dim=1)
                v_full = torch.cat([past_v, v_c], dim=1)
                full_len = k_full.shape[1]

                q_lens_c = torch.tensor(
                    [chunk_len] * b, dtype=torch.long, device=seq_lens.device)
                k_lens_c = torch.tensor(
                    [full_len] * b, dtype=torch.long, device=seq_lens.device)
                out_c = flash_attention(
                    q=q_c, k=k_full, v=v_full,
                    q_lens=q_lens_c, k_lens=k_lens_c,
                    window_size=self.window_size)
            else:
                # First chunk: standard bidirectional self-attention
                q_lens_c = torch.tensor(
                    [chunk_len] * b, dtype=torch.long, device=seq_lens.device)
                out_c = flash_attention(
                    q=q_c, k=k_c, v=v_c,
                    q_lens=q_lens_c, k_lens=q_lens_c,
                    window_size=self.window_size)

            output_chunks.append(out_c)
            past_k_list.append(k_c)
            past_v_list.append(v_c)

        # Reassemble and pad to original sequence length
        x = torch.cat(output_chunks, dim=1)
        if total_tokens < s:
            pad = x.new_zeros(b, s - total_tokens, x.shape[2], x.shape[3])
            x = torch.cat([x, pad], dim=1)

    elif use_cache and self._cache_k is not None and self._cache_valid_len > 0:
        # Concat cached K/V with current K/V
        cached_k = self._cache_k[:, :self._cache_valid_len]
        cached_v = self._cache_v[:, :self._cache_valid_len]
        k_full = torch.cat([cached_k, k[:, :seq_lens[0]]], dim=1)
        v_full = torch.cat([cached_v, v[:, :seq_lens[0]]], dim=1)
        full_len = k_full.shape[1]

        q_valid = q[:, :seq_lens[0]]
        k_lens = torch.tensor(
            [full_len] * b, dtype=torch.long, device=seq_lens.device)
        x = flash_attention(
            q=q_valid, k=k_full, v=v_full,
            k_lens=k_lens, window_size=self.window_size)
        valid_len = seq_lens[0].item() if seq_lens[0].dim() == 0 else int(seq_lens[0])
        if valid_len < s:
            pad = x.new_zeros(b, s - valid_len, x.shape[2], x.shape[3])
            x = torch.cat([x, pad], dim=1)
    else:
        x = flash_attention(
            q=q, k=k, v=v,
            k_lens=seq_lens, window_size=self.window_size)

    if use_cache == 'read_write' and self._cache_k is not None:
        self.append_to_cache(k[:, :seq_lens[0]], v[:, :seq_lens[0]])

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


# =============================================================================
# Patched forward: WanAttentionBlock
# =============================================================================


def block_causal_attn_block_forward(
    self, x, e, seq_lens, grid_sizes, freqs,
    context, context_lens, dit_cond_dict=None,
    frame_offset=0, use_cache=False,
    block_causal=False, chunk_size_tokens=0,
):
    """WanAttentionBlock.forward with block_causal pass-through."""
    import torch.nn.functional as torch_F

    assert e.dtype == torch.float32
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    # self-attention (with block_causal forwarded)
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
        seq_lens, grid_sizes, freqs,
        frame_offset=frame_offset, use_cache=use_cache,
        block_causal=block_causal, chunk_size_tokens=chunk_size_tokens)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

    # cam injection
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_hidden_states = self.cam_injector_layer2(
            torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
        c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
        cam_scale = self.cam_scale_layer(c2ws_hidden_states)
        cam_shift = self.cam_shift_layer(c2ws_hidden_states)
        x = (1.0 + cam_scale) * x + cam_shift

    # cross-attention & ffn
    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x

    x = cross_attn_ffn(x, context, context_lens, e)
    return x


# =============================================================================
# Patched forward: WanModel
# =============================================================================


def block_causal_model_forward(
    self, x, t, context, seq_len, y=None, dit_cond_dict=None,
    frame_offset=0, use_cache=False,
    block_causal=False, chunk_size_tokens=0,
):
    """WanModel.forward with block_causal + chunk_size_tokens kwargs."""
    from einops import rearrange

    if self.model_type == 'i2v':
        assert y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        from wan.modules.model import sinusoidal_embedding_1d
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # cam
    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i,
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.patch_size[0],
                c2=self.patch_size[1],
                c3=self.patch_size[2],
            ) for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch.nn.functional.silu(
                self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = (
            c2ws_plucker_emb + c2ws_hidden_states)

    # arguments — includes block_causal + chunk_size_tokens
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict,
        frame_offset=frame_offset,
        use_cache=use_cache,
        block_causal=block_causal,
        chunk_size_tokens=chunk_size_tokens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


# =============================================================================
# Entry point
# =============================================================================


def patch_model_for_block_causal(model):
    """Monkey-patch WanModel for block causal training.

    Call once after loading the model, before training. Replaces forward
    methods on self_attn, attention blocks, and the top-level model to
    accept block_causal + chunk_size_tokens kwargs.

    When block_causal=False (default), behavior is identical to original.

    Args:
        model: WanModel instance.
    """
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(
            block_causal_self_attn_forward, block.self_attn)
        block.forward = types.MethodType(
            block_causal_attn_block_forward, block)
    model.forward = types.MethodType(block_causal_model_forward, model)
