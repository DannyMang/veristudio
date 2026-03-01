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


# =============================================================================
# Patched generate_chunk: remove per-step empty_cache + optional no-CFG
# =============================================================================


def optimized_generate_chunk(self, img, prompt, c2ws, intrinsics,
                             frame_num=17, shift=5.0, seed=42, cfg_scale=None):
    """Optimized generate_chunk for WanI2VCausal.

    Changes from original:
      1. torch.cuda.empty_cache() moved outside denoising loop (saves ~0.5-1s).
      2. cfg_scale parameter: when <= 1.0, skips unconditional forward pass
         entirely (2x speedup for distilled models that don't need CFG).
         Default None = use self.guide_scale (backward compatible).
    """
    import torchvision.transforms.functional as TF_local
    import torchvision.transforms.functional as TF

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from tqdm import tqdm

    F_chunk = min(frame_num, ((len(c2ws) - 1) // 4) * 4 + 1)
    c2ws = c2ws[:F_chunk]
    intrinsics = intrinsics[:F_chunk]

    lat_f = (F_chunk - 1) // self.vae_stride[0] + 1
    max_seq_len = lat_f * self._tokens_per_lat_frame

    self._ensure_cache(lat_f)

    # Text encoding (cached)
    context, context_null = self._encode_text(prompt)

    # Plucker embeddings (chunk-local)
    dit_cond_dict = self._compute_plucker(c2ws, intrinsics, lat_f)

    # Prepare image conditioning
    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
    msk = torch.ones(1, F_chunk, self.lat_h, self.lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
    msk = msk.transpose(1, 2)[0]

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img_t[None].cpu(), size=(self.h, self.w),
                mode='bicubic').transpose(0, 1),
            torch.zeros(3, F_chunk - 1, self.h, self.w)
        ], dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    # Noise
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16, lat_f, self.lat_h, self.lat_w,
        dtype=torch.float32, generator=seed_g, device=self.device)

    skip_cfg = cfg_scale is not None and cfg_scale <= 1.0

    with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
        # Scheduler
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(self.sampling_steps, device=self.device, shift=shift)

        # Denoise with MoE routing + CFG
        # KV cache only used when single_expert (post-trained causal model)
        use_kv = self._cache_initialized
        latent = noise
        base_args = {
            'seq_len': max_seq_len,
            'y': [y],
            'dit_cond_dict': dit_cond_dict,
            'frame_offset': self.frame_offset,
            'use_cache': 'read_only' if use_kv else False,
        }

        for _, t in enumerate(tqdm(scheduler.timesteps, desc='denoise', leave=False)):
            latent_model_input = [latent.to(self.device)]
            timestep = torch.stack([t]).to(self.device)

            # Select expert based on timestep (handles offloading)
            model = self._get_model_for_timestep(t)

            # Conditional forward pass
            noise_pred_cond = model(
                latent_model_input, t=timestep,
                context=[context[0]], **base_args)[0]

            if skip_cfg:
                # No CFG: use conditional prediction directly
                noise_pred = noise_pred_cond
            else:
                # Unconditional forward pass (CFG)
                scale = self.guide_scale[1] if t.item() >= self.boundary else self.guide_scale[0]
                noise_pred_uncond = model(
                    latent_model_input, t=timestep,
                    context=context_null, **base_args)[0]
                noise_pred = noise_pred_uncond + scale * (
                    noise_pred_cond - noise_pred_uncond)
                del noise_pred_uncond

            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            del noise_pred_cond, noise_pred

        torch.cuda.empty_cache()

        # Phase 2: Cache-fill (only for single-expert causal mode)
        if use_kv:
            cache_model = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
            t_zero = torch.zeros(1, device=self.device)
            _ = cache_model(
                [latent.to(self.device)], t=t_zero,
                context=[context[0]],
                seq_len=max_seq_len, y=[y],
                dit_cond_dict=dit_cond_dict,
                frame_offset=self.frame_offset,
                use_cache='read_write')

        # Advance global frame offset
        self.frame_offset += lat_f

        # VAE decode
        videos = self.vae.decode([latent])

    video = videos[0]  # (C, N, H, W)

    # Extract last frame as PIL
    last = video[:, -1, :, :]
    last = ((last + 1.0) / 2.0).clamp(0, 1).cpu()
    last_pil = TF_local.to_pil_image(last)

    del noise, scheduler
    return video, last_pil


def patch_generate_chunk(causal_model):
    """Monkey-patch WanI2VCausal.generate_chunk with optimized version.

    Optimizations:
      1. Moves torch.cuda.empty_cache() outside denoising loop (~0.5-1s saved).
      2. Adds cfg_scale parameter — set to 1.0 to skip CFG (2x speedup).

    Args:
        causal_model: WanI2VCausal instance.
    """
    causal_model.generate_chunk = types.MethodType(
        optimized_generate_chunk, causal_model)
