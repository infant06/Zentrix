# FlexLoad Cache Mismatch Bug

## Overview
During the implementation of `FlexLoad` for `zen-core`, a severe shape mismatch bug occurred when generating attention masks for vision-language models (e.g., Qwen2-VL). 
The error manifested as a shape mismatch in `broadcast_add` within `Sdpa::run_attention`:

```
shape mismatch in broadcast_add, lhs: [1, 14, 8, 8], rhs: [8, 16]
```

## Root Cause Analysis
The bug occurred due to a discrepancy between the **Global KV Cache** and the **Local Preallocated Layer Caches** used by `FlexLoad`.

1. **Dummy Cache Initialization**: When streaming a layer from disk, `FlexLoad` generates a "dummy" local layer cache initialized with `current_seq_len: 0` using `zeros_like()` for memory efficiency.
2. **Mask Generation Uses Global Cache**: The global `ModelForwardContext` creates the attention mask (`ctx.mask_cache(cache)`) by passing a reference to the *global* `KvCache`. The global cache correctly retains the actual token history and reports a `past_kv_len` (e.g., `8`).
3. **Mismatched Lengths**: `CausalMasker` uses `past_kv_len: 8` to generate an `[8, 16]` attention mask (`tgt_len=8`, `offset=16`).
4. **Local Cache Overwrite**: Inside `layer.forward(...)`, the local dummy cache is used. Since its `current_seq_len` is artificially set to `0`, `KvCache::append` blindly appends the 8 new tokens starting at index 0. This results in the Key/Value tensors having a sequence length of `8` instead of `16`.
5. **Crash in Attention**: The attention scores (`lhs = q * k.t()`) resolve to shape `[1, 14, 8, 8]`. `run_attention_noflash` attempts to `broadcast_add` the `[8, 16]` mask onto the `[8, 8]` scores, triggering a fatal shape mismatch.

## Temporary Workaround
Current generative models (like `Qwen2.5`) avoid this bug because they mutate the global cache directly via references (`self.model.cache.normal_mut()`), bypassing the dummy `local_layer_caches`. 

## Long-term Fix
When introducing support for Llama, Mistral, DeepSeek, or other MoE models under FlexLoad or PartialOffload, ensure that:
1. Preallocated streamed caches correctly track and resume from `global_cache.current_seq_len()`.
2. Do not reset `current_seq_len: 0` unless it truly is the first prompt chunk.
3. Decouple attention mask creation from relying on implicitly aligned cache instances, or ensure local layer cache mutations synchronize perfectly with the global context.
