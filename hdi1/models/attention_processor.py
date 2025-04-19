from typing import Optional
import torch
# Adjust relative import based on actual structure
try:
    from .attention import HiDreamAttention
except ImportError:
     # Fallback if running script standalone or structure differs
    print("Could not import HiDreamAttention via relative path in attention_processor.py")
    import torch.nn as nn
    class HiDreamAttention(nn.Module): pass # Dummy placeholder

ATTN_FUNC_BACKEND = None
import einops
try:
    try:
        from flash_attn_interface import flash_attn_func
        ATTN_FUNC_BACKEND = "FLASH_ATTN_3"
        print("Using Flash Attention 3 backend.")
    except ImportError:
        from flash_attn import flash_attn_func
        ATTN_FUNC_BACKEND = "FLASH_ATTN_2"
        print("Using Flash Attention 2 backend.")
except ImportError:
    import torch.nn.functional as F
    ATTN_FUNC_BACKEND = "VANILLA"
    print("Flash Attention not found. Using vanilla PyTorch attention backend.")


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # <<< Ensure inputs are on the SAME device before operating >>> ### ADDED ###
    target_device = xq.device
    xk = xk.to(target_device)
    freqs_cis = freqs_cis.to(target_device)

    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # Ensure freqs_cis matches shape for broadcasting if needed, and is float
    freqs_cis_ = freqs_cis.float().reshape(*freqs_cis.shape[:-1], -1, 2, 2) # Reshape to match xq_/xk_ last dims

    # Perform operations using reshaped tensors
    xq_r, xq_i = xq_[..., 0:1], xq_[..., 1:2] # Keep dimensions for broadcasting
    xk_r, xk_i = xk_[..., 0:1], xk_[..., 1:2]
    freqs_cos = freqs_cis_[..., 0:1, 0:1] # Slice cos part, keep dims
    freqs_sin = freqs_cis_[..., 0:1, 1:2] # Slice sin part, keep dims

    xq_out_r = freqs_cos * xq_r - freqs_sin * xq_i
    xq_out_i = freqs_sin * xq_r + freqs_cos * xq_i

    xk_out_r = freqs_cos * xk_r - freqs_sin * xk_i
    xk_out_i = freqs_sin * xk_r + freqs_cos * xk_i

    # Combine real and imaginary parts
    xq_out = torch.cat([xq_out_r, xq_out_i], dim=-1).reshape(*xq.shape)
    xk_out = torch.cat([xk_out_r, xk_out_i], dim=-1).reshape(*xk.shape)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    target_device = query.device
    key = key.to(target_device)
    value = value.to(target_device)
    # print(f"[attention fn] Q:{query.device} K:{key.device} V:{value.device}") # Debug log

    if ATTN_FUNC_BACKEND == "FLASH_ATTN_3":
        # FlashAttn 3 interface might differ, check documentation if used
        # Assuming flash_attn_func handles dtype conversions internally if needed
        hidden_states = flash_attn_func(query.to(target_device), key.to(target_device), value.to(target_device), causal=False)[0] # Re-check determinism flag if needed
    elif ATTN_FUNC_BACKEND == "FLASH_ATTN_2":
        # Ensure inputs are contiguous and compatible dtype (e.g., float16, bfloat16)
        query_fa = query.contiguous()
        key_fa = key.contiguous()
        value_fa = value.contiguous()
        # print(f"[attention fn] FlashAttn2 Input Dtypes - Q:{query_fa.dtype}, K:{key_fa.dtype}, V:{value_fa.dtype}")
        try:
            hidden_states = flash_attn_func(query_fa, key_fa, value_fa, dropout_p=0.0, causal=False)
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}. Falling back to PyTorch SDPA.")
            # Fallback to PyTorch SDPA
            query = einops.rearrange(query, 'b s h d -> b h s d')
            key = einops.rearrange(key, 'b s h d -> b h s d')
            value = einops.rearrange(value, 'b s h d -> b h s d')
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = einops.rearrange(hidden_states, 'b h s d -> b s (h d)') # Flatten head and dim back
            hidden_states = hidden_states.to(query.dtype) # Ensure original dtype
            return hidden_states # Return early after SDPA
    elif ATTN_FUNC_BACKEND == "VANILLA":
        # Use einops for transpose: b s (h d) -> b h s d
        query = einops.rearrange(query, 'b s h d -> b h s d')
        key = einops.rearrange(key, 'b s h d -> b h s d')
        value = einops.rearrange(value, 'b s h d -> b h s d')
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # Use einops for transpose: b h s d -> b s (h d)
        hidden_states = einops.rearrange(hidden_states, 'b h s d -> b s (h d)') # Flatten head and dim back
    else:
        raise RuntimeError(f"Unknown attention backend: {ATTN_FUNC_BACKEND}")

    # If not VANILLA SDPA fallback, flatten the heads*dim dimension
    if ATTN_FUNC_BACKEND != "VANILLA" or 'hidden_states' not in locals(): # Ensure hidden_states exists if flashattn fails
         hidden_states = hidden_states.flatten(-2) # Flatten head and dim: b s (h d)

    hidden_states = hidden_states.to(query.dtype) # Ensure output dtype matches query
    return hidden_states


class HiDreamAttnProcessor_flashattn:
    """Attention processor with device checks and FlashAttention support."""

    def __call__(
        self,
        attn: HiDreamAttention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None, # Might be CPU if T5 wasn't moved properly earlier
        rope: torch.FloatTensor = None, # Should be target device
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # Determine target device from the attention layer itself
        target_device = attn.to_q.weight.device
        dtype = image_tokens.dtype # Use image tokens dtype as reference
        batch_size = image_tokens.shape[0]
        # print(f"[AttnProc] Target device: {target_device}, Input image_tokens device: {image_tokens.device}")

        # Ensure main input is on target device
        image_tokens = image_tokens.to(target_device)

        # --- Projections (Layers are on target_device) ---
        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        # Check devices after initial projections
        # print(f"[AttnProc] Q_i: {query_i.device}, K_i: {key_i.device}, V_i: {value_i.device}")

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)

        if image_tokens_masks is not None:
            image_tokens_masks = image_tokens_masks.to(target_device) # Ensure mask is on device
            # Apply mask safely
            key_i = torch.where(image_tokens_masks.view(batch_size, -1, 1, 1), key_i, torch.zeros_like(key_i))


        if not attn.single:
            # <<< Ensure text_tokens is moved >>> ### ADDED Check ###
            if text_tokens is None: raise ValueError("text_tokens cannot be None for non-single attention")
            text_tokens = text_tokens.to(target_device, dtype=dtype) # Ensure device and dtype
            # print(f"[AttnProc] text_tokens device after move: {text_tokens.device}")

            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            # Check devices
            # print(f"[AttnProc] Q_t: {query_t.device}, K_t: {key_t.device}, V_t: {value_t.device}")

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]

            # Ensure concatenation operands are on the same device (target_device)
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        # Ensure rope is on the correct device and dtype
        if rope is not None:
            rope = rope.to(target_device, dtype=query.dtype) # Match query dtype
            # print(f"[AttnProc] rope device: {rope.device}")

        # --- RoPE Application ---
        # Check devices before apply_rope
        # print(f"[AttnProc] Before RoPE - Q: {query.device}, K: {key.device}, rope: {rope.device if rope is not None else 'None'}")
        if rope is not None:
            # The check for rope shape vs query shape might need adjustment based on how rope is structured
            # Assuming apply_rope handles device internally now based on its first arg (query)
             if query.shape[-1] == rope.shape[-3] * 2: # Original check
                query, key = apply_rope(query, key, rope)
             else: # Original chunking logic, ensure inputs to apply_rope are correct device
                query_1, query_2 = query.chunk(2, dim=-1)
                key_1, key_2 = key.chunk(2, dim=-1)
                # Pass rope slice if needed, ensure device
                rope_slice = rope # Assuming rope applies to first half? Adjust if needed.
                query_1, key_1 = apply_rope(query_1.to(target_device), key_1.to(target_device), rope_slice.to(target_device))
                query = torch.cat([query_1, query_2.to(target_device)], dim=-1)
                key = torch.cat([key_1, key_2.to(target_device)], dim=-1)
        # print(f"[AttnProc] After RoPE - Q: {query.device}, K: {key.device}")


        # --- Attention Calculation ---
        # Ensure all inputs to attention function are on the target device
        query = query.to(target_device)
        key = key.to(target_device)
        value = value.to(target_device)
        # print(f"[AttnProc] Before Attention Fn - Q: {query.device}, K: {key.device}, V: {value.device}")

        # Call the backend-specific attention function
        hidden_states = attention(query, key, value)

        # print(f"[AttnProc] After Attention Fn - hidden_states: {hidden_states.device}")

        # --- Output Projection ---
        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            # Ensure inputs to linear layers are correct device
            hidden_states_i = attn.to_out(hidden_states_i.to(target_device))
            hidden_states_t = attn.to_out_t(hidden_states_t.to(target_device))
            # print(f"[AttnProc] Final output devices - i: {hidden_states_i.device}, t: {hidden_states_t.device}")
            return hidden_states_i.to(dtype=dtype), hidden_states_t.to(dtype=dtype) # Cast back to original dtype
        else:
            hidden_states = attn.to_out(hidden_states.to(target_device))
            # print(f"[AttnProc] Final output device - single: {hidden_states.device}")
            return hidden_states.to(dtype=dtype) # Cast back to original dtype
