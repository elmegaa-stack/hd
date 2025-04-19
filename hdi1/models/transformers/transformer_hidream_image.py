from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import einops
from einops import repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
# Adjust relative imports based on the actual location of this file relative to embeddings etc.
# Assuming it's in hd/hdi1/models/transformers/
from ..embeddings import PatchEmbed, PooledEmbed, TimestepEmbed, EmbedND, OutEmbed
from ..attention import HiDreamAttention, FeedForwardSwiGLU
from ..attention_processor import HiDreamAttnProcessor_flashattn
from ..moe import MOEFeedForwardSwiGLU

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states

class BlockType:
    TransformerBlock = 1
    SingleTransformerBlock = 2

@maybe_allow_in_graph
class HiDreamImageSingleTransformerBlock(nn.Module):
    # --- Keep original class definition ---
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = True
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim, # Standard SwiGLU hidden dim, adjust if needed
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim) # Non-MoE version

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None, # Not used in single block
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,

    ) -> torch.FloatTensor:
        # Determine target device and dtype from main input
        target_device = image_tokens.device
        wtype = image_tokens.dtype # Use input dtype

        # Ensure adaln_input is on the correct device before processing
        adaln_input = adaln_input.to(target_device, dtype=wtype)
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)

        # 1. Attention
        # Ensure rope is on the correct device if provided
        if rope is not None: rope = rope.to(target_device, dtype=wtype)

        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype) # Norm doesn't change device
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_image_tokens.to(target_device), # Ensure input to attention is on target device
            image_tokens_masks.to(target_device) if image_tokens_masks is not None else None, # Ensure mask on target device
            rope = rope, # Already moved
        )
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(target_device)) # Ensure input to FF is on target device
        image_tokens = ff_output_i + image_tokens
        return image_tokens


@maybe_allow_in_graph
class HiDreamImageTransformerBlock(nn.Module):
    # --- Keep original class definition ---
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 12 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.norm1_t = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = False
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim, # Adjust if needed
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim)
        self.norm3_t = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.ff_t = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim) # Non-MoE version for text? Check architecture

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None, # Might come from CPU/mixed devices
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Determine target device and dtype from main input
        target_device = image_tokens.device
        wtype = image_tokens.dtype # Use input dtype

        # Ensure adaln_input is on the correct device before processing
        adaln_input = adaln_input.to(target_device, dtype=wtype)
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)

        # 1. MM-Attention
        # Ensure rope is on the correct device if provided
        if rope is not None: rope = rope.to(target_device, dtype=wtype)

        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i

        # <<< Ensure text_tokens is on the correct device >>> ### ADDED ###
        norm_text_tokens = self.norm1_t(text_tokens.to(target_device)).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        # Ensure inputs to attention are on target device
        attn_output_i, attn_output_t = self.attn1(
            hidden_states=norm_image_tokens.to(target_device),
            attention_mask=image_tokens_masks.to(target_device) if image_tokens_masks is not None else None,
            encoder_hidden_states=norm_text_tokens.to(target_device), # Ensure encoder states are on device
            rope = rope,
        )

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens # text_tokens is now updated on target_device

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(target_device)) # Ensure input to FF is on target device
        image_tokens = ff_output_i + image_tokens

        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype) # text_tokens is now on target_device
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens.to(target_device)) # Ensure input to FF is on target device
        text_tokens = ff_output_t + text_tokens

        return image_tokens, text_tokens


@maybe_allow_in_graph
class HiDreamImageBlock(nn.Module):
     # --- Keep original class definition ---
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        block_type: BlockType = BlockType.TransformerBlock,
    ):
        super().__init__()
        block_classes = {
            BlockType.TransformerBlock: HiDreamImageTransformerBlock,
            BlockType.SingleTransformerBlock: HiDreamImageSingleTransformerBlock,
        }
        self.block = block_classes[block_type](
            dim,
            num_attention_heads,
            attention_head_dim,
            num_routed_experts,
            num_activated_experts
        )

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None, # This might be on CPU/mixed
        adaln_input: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        # Get target device from main image input
        target_device = image_tokens.device
        # Ensure all inputs passed to the sub-block are on the target device
        if text_tokens is not None: text_tokens = text_tokens.to(target_device)
        if adaln_input is not None: adaln_input = adaln_input.to(target_device)
        if rope is not None: rope = rope.to(target_device)
        if image_tokens_masks is not None: image_tokens_masks = image_tokens_masks.to(target_device)

        return self.block(
            image_tokens.to(target_device), # Already should be, but ensure
            image_tokens_masks,
            text_tokens,
            adaln_input,
            rope,
        )


class HiDreamImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    # --- Keep __init__ and other methods the same ---
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.llama_layers = llama_layers

        self.t_embedder = TimestepEmbed(self.inner_dim)
        self.p_embedder = PooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = PatchEmbed(
            patch_size = patch_size,
            in_channels = in_channels,
            out_channels = self.inner_dim,
        )
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.config.num_attention_heads,
                    attention_head_dim = self.config.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.TransformerBlock
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.config.num_attention_heads,
                    attention_head_dim = self.config.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.SingleTransformerBlock
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.final_layer = OutEmbed(self.inner_dim, patch_size, self.out_channels)

        caption_channels = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features = caption_channel, hidden_size = self.inner_dim))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        return timesteps

    def unpatchify(self, x: torch.Tensor, img_sizes: List[Tuple[int, int]], is_training: bool) -> List[torch.Tensor]:
        if is_training:
            x = einops.rearrange(x, 'B S (p1 p2 C) -> B C S (p1 p2)', p1=self.config.patch_size, p2=self.config.patch_size)
        else:
            x_arr = []
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                # Ensure slicing/reshaping inputs are on correct device
                current_x = x[i, :pH*pW].reshape(1, pH, pW, -1).to(x.device)
                x_arr.append(
                    einops.rearrange(current_x, 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                        p1=self.config.patch_size, p2=self.config.patch_size)
                )
            x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, x, max_seq, img_sizes=None):
        pz2 = self.config.patch_size * self.config.patch_size
        if isinstance(x, torch.Tensor):
            B, C = x.shape[0], x.shape[1]
            device = x.device
            dtype = x.dtype
        else: # Assuming list of tensors
            B, C = len(x), x[0].shape[0]
            device = x[0].device # Assume all list elements on same device initially
            dtype = x[0].dtype
            x = torch.stack(x) # Convert list to tensor

        # Ensure masks created on the correct device
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes):
                x_masks[i, 0:img_size[0] * img_size[1]] = 1
            # Ensure x is on device before rearrange
            x = einops.rearrange(x.to(device), 'B C S p -> B S (p C)', p=pz2)
        elif isinstance(x, torch.Tensor):
            pH, pW = x.shape[-2] // self.config.patch_size, x.shape[-1] // self.config.patch_size
            x = einops.rearrange(x.to(device), 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.config.patch_size, p2=self.config.patch_size)
            img_sizes = [[pH, pW]] * B
            x_masks = None # Mask is not used if img_sizes wasn't provided initially
        else:
            raise NotImplementedError # Should be caught by earlier isinstance check
        return x, x_masks, img_sizes

    # --- MODIFIED forward Method ---
    def forward(
        self,
        hidden_states: torch.Tensor, # Assume this is on cuda:0 (target device)
        timesteps: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None, # List [T5(cuda:0), Llama(cuda:0)] after _encode_prompt fix
        pooled_embeds: torch.Tensor = None, # Should be cuda:0 after _encode_prompt fix
        img_sizes: Optional[List[Tuple[int, int]]] = None, # Provided by pipeline __call__
        img_ids: Optional[torch.Tensor] = None, # Provided by pipeline __call__
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        # Get the target device from the main input (where this model instance lives)
        target_device = hidden_states.device
        hidden_states_type = hidden_states.dtype

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # spatial forward
        batch_size = hidden_states.shape[0]

        # 0. time
        # Ensure timesteps tensor is on the target device
        timesteps = self.expand_timesteps(timesteps, batch_size, target_device)
        # t_embedder is on target_device
        time_embed = self.t_embedder(timesteps, hidden_states_type) # Pass correct dtype
        # Ensure pooled_embeds is on target device and correct dtype
        pooled_embeds = pooled_embeds.to(target_device, dtype=hidden_states_type)
        # p_embedder is on target_device
        pooled_embed = self.p_embedder(pooled_embeds)
        # Combine embeddings -> should be on target_device
        adaln_input = time_embed + pooled_embed

        # 1. Patchify and Embeddings
        # Ensure input hidden_states is on target_device before patchify
        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states.to(target_device), self.max_seq, img_sizes)
        # Ensure masks are on target_device
        if image_tokens_masks is not None: image_tokens_masks = image_tokens_masks.to(target_device)

        # Create img_ids if needed, ensuring on target_device
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=target_device, dtype=torch.int64) # Use target_device
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=target_device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=target_device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        else: # Ensure provided img_ids is on target device
             img_ids = img_ids.to(target_device, dtype=torch.int64)

        # Apply patch embedding (x_embedder is on target_device)
        hidden_states = self.x_embedder(hidden_states)

        # Ensure encoder_hidden_states list elements are on target device ### ADDED Check ###
        # We expect list [T5_embeds, Llama_embeds_stack] from _encode_prompt
        if not isinstance(encoder_hidden_states, (list, tuple)) or len(encoder_hidden_states) < 2:
             raise ValueError("encoder_hidden_states must be a list/tuple of length at least 2.")

        T5_encoder_hidden_states = encoder_hidden_states[0].to(target_device, dtype=hidden_states_type)
        _llama_hidden_states_stack = encoder_hidden_states[-1].to(target_device, dtype=hidden_states_type)

        # Select specific llama layers
        # Ensure the selected layers stay on the correct device
        encoder_hidden_states = [_llama_hidden_states_stack[k].to(target_device, dtype=hidden_states_type) for k in self.llama_layers]

        # Project captions (projections are on target_device)
        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                # Ensure input is correct device/dtype before projection
                enc_hidden_state_proj = self.caption_projection[i](enc_hidden_state.to(target_device, dtype=hidden_states_type))
                enc_hidden_state_proj = enc_hidden_state_proj.view(batch_size, -1, self.inner_dim)
                new_encoder_hidden_states.append(enc_hidden_state_proj)
            encoder_hidden_states = new_encoder_hidden_states # Now contains projected Llama layers

            # Ensure T5 input is correct device/dtype before projection
            T5_encoder_hidden_states_proj = self.caption_projection[-1](T5_encoder_hidden_states.to(target_device, dtype=hidden_states_type))
            T5_encoder_hidden_states_proj = T5_encoder_hidden_states_proj.view(batch_size, -1, self.inner_dim)
            # Append projected T5 state; encoder_hidden_states is now [proj_llama..., proj_T5]
            encoder_hidden_states.append(T5_encoder_hidden_states_proj)

        # Create text IDs tensor on target device
        # Ensure shapes used are from tensors guaranteed to be on target_device
        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1], # Shapes of projected tensors
            3,
            device=target_device, dtype=img_ids.dtype
        )
        # Ensure img_ids is on target device before cat
        ids = torch.cat((img_ids.to(target_device), txt_ids), dim=1)
        # Calculate RoPE, ensure output is on target device
        # pe_embedder is on target_device
        rope = self.pe_embedder(ids).to(target_device, dtype=hidden_states_type)


        # 2. Blocks
        block_id = 0
        # Ensure initial states are on target device
        # Use projected T5 (-1) and last projected Llama (-2)
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1).to(target_device, dtype=hidden_states_type)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]

        # Ensure hidden_states is on target_device before loop
        hidden_states = hidden_states.to(target_device)

        for bid, block in enumerate(self.double_stream_blocks):
            # Ensure ALL inputs to block are on target device
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id].to(target_device, dtype=hidden_states_type)
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1).to(target_device, dtype=hidden_states_type)
            hidden_states_input = hidden_states # Already on target device
            adaln_input_block = adaln_input.to(target_device, dtype=hidden_states_type) # Ensure again
            rope_block = rope.to(target_device, dtype=hidden_states_type) # Ensure again
            image_tokens_masks_block = image_tokens_masks.to(target_device) if image_tokens_masks is not None else None # Ensure again

            # Block execution (assumes internal ops respect device)
            if self.training and self.gradient_checkpointing:
                 raise NotImplementedError("Gradient checkpointing not fully checked for device placement")
                 # ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                 # hidden_states, initial_encoder_hidden_states_out = torch.utils.checkpoint.checkpoint(...)
            else:
                hidden_states, initial_encoder_hidden_states_out = block(
                    image_tokens = hidden_states_input,
                    image_tokens_masks = image_tokens_masks_block,
                    text_tokens = cur_encoder_hidden_states,
                    adaln_input = adaln_input_block,
                    rope = rope_block,
                )
            # Ensure output is on target device before slicing/next iteration
            initial_encoder_hidden_states = initial_encoder_hidden_states_out[:, :initial_encoder_hidden_states_seq_len].to(target_device, dtype=hidden_states_type)
            hidden_states = hidden_states.to(target_device) # Ensure main hidden_states stays on device
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        # Ensure inputs for concatenation are on target device
        hidden_states = torch.cat([hidden_states.to(target_device), initial_encoder_hidden_states.to(target_device)], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]

        if image_tokens_masks is not None:
            # Ensure mask concatenation operands are on target device
            # Use shape from last loop iteration's cur_llama31_encoder_hidden_states
            last_cur_llama_shape_1 = encoder_hidden_states[block_id-1].shape[1] # block_id was incremented
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + last_cur_llama_shape_1),
                device=target_device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks.to(target_device), encoder_attention_mask_ones], dim=1)


        for bid, block in enumerate(self.single_stream_blocks):
             # Ensure ALL inputs to block are on target device
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id].to(target_device, dtype=hidden_states_type) # Use correct block_id
            hidden_states_input = torch.cat([hidden_states.to(target_device), cur_llama31_encoder_hidden_states], dim=1)
            adaln_input_block = adaln_input.to(target_device, dtype=hidden_states_type)
            rope_block = rope.to(target_device, dtype=hidden_states_type)
            image_tokens_masks_block = image_tokens_masks.to(target_device) if image_tokens_masks is not None else None

            if self.training and self.gradient_checkpointing:
                 raise NotImplementedError("Gradient checkpointing not fully checked for device placement")
                 # ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                 # hidden_states = torch.utils.checkpoint.checkpoint(...)
            else:
                # Pass only relevant inputs to SingleTransformerBlock
                hidden_states = block(
                    image_tokens = hidden_states_input,
                    image_tokens_masks = image_tokens_masks_block,
                    text_tokens = None, # Single block doesn't use this
                    adaln_input = adaln_input_block,
                    rope = rope_block,
                )
            # Slice output, ensure device
            hidden_states = hidden_states[:, :hidden_states_seq_len].to(target_device, dtype=hidden_states_type)
            block_id += 1 # Increment block_id for next llama layer index

        # Final processing
        hidden_states = hidden_states[:, :image_tokens_seq_len, ...] # Slice final image tokens
        # Ensure inputs to final layer are on target device
        output = self.final_layer(hidden_states.to(target_device), adaln_input.to(target_device))
        output = self.unpatchify(output, img_sizes, self.training) # Unpatchify handles device internally based on input
        if image_tokens_masks is not None:
            image_tokens_masks = image_tokens_masks[:, :image_tokens_seq_len] # Slice mask

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, image_tokens_masks)
        return Transformer2DModelOutput(sample=output, mask=image_tokens_masks)


    # --- Keep other methods like _set_gradient_checkpointing, unpatchify, patchify, expand_timesteps the same ---
