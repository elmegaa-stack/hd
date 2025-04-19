import math
import torch
from torch import nn
import torch.nn.functional as F
# Adjust relative import based on actual structure
try:
    from .attention import FeedForwardSwiGLU
except ImportError:
    # Fallback if running script standalone or structure differs
    print("Could not import FeedForwardSwiGLU via relative path in moe.py")
    # You might need to define FeedForwardSwiGLU here or adjust the import path
    class FeedForwardSwiGLU(nn.Module): # Dummy placeholder if import fails
         def __init__(self, *args, **kwargs): super().__init__(); self.linear=nn.Linear(1,1)
         def forward(self,x): return x


# Assume torch.distributed is available; handle if not needed/present
from torch.distributed.nn.functional import all_gather as torch_all_gather
IS_DISTRIBUTED = torch.distributed.is_available() and torch.distributed.is_initialized()

_LOAD_BALANCING_LOSS = []
def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)

def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS

def batched_load_balancing_loss():
    aux_losses_arr = get_load_balancing_loss()
    if not aux_losses_arr: return 0.0 # Return 0 if no loss saved

    alpha = aux_losses_arr[0][-1]
    Pi = torch.stack([ent[1] for ent in aux_losses_arr], dim=0)
    fi = torch.stack([ent[2] for ent in aux_losses_arr], dim=0)

    if IS_DISTRIBUTED:
        # If distributed training is used, gather 'fi' across devices
        fi_list = torch_all_gather(fi)
        fi = torch.stack(fi_list, 0).mean(0)
    else:
        # Otherwise, just use the local 'fi'
        pass

    aux_loss = (Pi * fi).sum(-1).mean() * alpha
    return aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    # --- Keep original class definition ---
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        target_device = hidden_states.device # Get device from input
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        # Ensure weight is on same device for F.linear
        logits = F.linear(hidden_states, self.weight.to(target_device), None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        # (Keep original aux loss calculation logic)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=target_device) # Use target_device
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=target_device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
                save_load_balancing_loss((aux_loss, Pi, fi, self.alpha))
        else:
            aux_loss = None
        # Ensure outputs are on the input device
        return topk_idx.to(target_device), topk_weight.to(target_device), aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    # --- Keep original __init__ ---
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
    ):
        super().__init__()
        # Adjust hidden_dim calculation if needed based on original model's SwiGLU implementation
        ffn_hidden_dim = int(2 * (hidden_dim // 2) / 3) # Example SwiGLU calculation
        # Or just use hidden_dim directly if that's how FeedForwardSwiGLU expects it
        # ffn_hidden_dim = hidden_dim

        self.shared_experts = FeedForwardSwiGLU(dim, ffn_hidden_dim // 2) # Check hidden dim for shared
        self.experts = nn.ModuleList([FeedForwardSwiGLU(dim, ffn_hidden_dim) for i in range(num_routed_experts)])
        self.gate = MoEGate(
            embed_dim = dim,
            num_routed_experts = num_routed_experts,
            num_activated_experts = num_activated_experts
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        target_device = x.device # Get device from input
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x) # Gate should return tensors on target_device
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # (Keep original training logic - might need device checks too)
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype) # Defaults to device of x
            for i, expert in enumerate(self.experts):
                # Ensure expert runs on correct device (should be target_device if model moved correctly)
                expert_device = next(expert.parameters()).device
                # Get indices for this expert
                expert_mask = (flat_topk_idx == i)
                if expert_mask.any():
                    # Ensure inputs/indices are on the expert's device
                     y[expert_mask] = expert(x[expert_mask].to(expert_device)).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype) # Ensure final shape and type
            # Handle aux_loss if needed (omitted for brevity)
        else:
             # Use modified moe_infer with device checks
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # Ensure shared expert input and output are on target device
        y = y + self.shared_experts(identity.to(target_device)).to(target_device)
        return y.to(target_device) # Ensure final output device

    # ### MODIFIED moe_infer with device checks ###
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        target_device = x.device # Get device from input
        expert_cache = torch.zeros_like(x) # Creates on target_device

        # Ensure indices and weights are on the correct device
        flat_expert_indices = flat_expert_indices.to(target_device)
        flat_expert_weights = flat_expert_weights.to(target_device)

        # print(f"[moe_infer] x device: {x.device}, flat_expert_indices device: {flat_expert_indices.device}")
        idxs = flat_expert_indices.argsort()
        # print(f"[moe_infer] idxs device: {idxs.device}")

        # bincount often returns CPU, but its result is used in Python loop range, not directly in CUDA ops.
        # This should be okay as long as indices used later (exp_token_idx) are derived from CUDA tensors.
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        token_idxs = idxs // self.num_activated_experts # Should be target_device
        # print(f"[moe_infer] token_idxs device: {token_idxs.device}")

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i] # Expert should be on target_device if model loaded correctly
            expert_device = next(expert.parameters()).device # Get actual expert device
            exp_token_idx = token_idxs[start_idx:end_idx].to(target_device) # Ensure index slice is on target

            # <<<--- Check devices right before indexing --- >>>
            if not (x.device == exp_token_idx.device):
                 print(f"WARNING [moe_infer] Device mismatch before expert indexing! x: {x.device}, exp_token_idx: {exp_token_idx.device}. Moving index.")
                 exp_token_idx = exp_token_idx.to(x.device) # Move index to match data

            expert_tokens = x[exp_token_idx] # Indexing happens here

            # Ensure input to expert matches expert's actual device
            expert_out = expert(expert_tokens.to(expert_device))

            # Ensure weights are on correct device before mul_
            expert_weights = flat_expert_weights[idxs[start_idx:end_idx]].to(expert_out.device)
            expert_out.mul_(expert_weights)

            # Ensure all inputs to scatter_reduce_ are on the same device (target_device)
            exp_token_idx_scatter = exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]).to(target_device)
            target_scatter_device = expert_cache.device # Should be target_device
            expert_out_scatter = expert_out.to(target_scatter_device) # Move output to target device for scatter

            # <<<--- Check devices right before scatter_reduce_ --- >>>
            if not (exp_token_idx_scatter.device == target_scatter_device):
                 print(f"WARNING [moe_infer] Index device mismatch for scatter_reduce! index: {exp_token_idx_scatter.device}, target: {target_scatter_device}. Moving index.")
                 exp_token_idx_scatter = exp_token_idx_scatter.to(target_scatter_device)
            if not (expert_out_scatter.device == target_scatter_device):
                 print(f"WARNING [moe_infer] Source device mismatch for scatter_reduce! source: {expert_out_scatter.device}, target: {target_scatter_device}. Moving source.")
                 expert_out_scatter = expert_out_scatter.to(target_scatter_device)

            # Use include_self=False if available (PyTorch >= 1.11 approx, depends on exact scatter_reduce impl)
            try:
                 expert_cache.scatter_reduce_(0, exp_token_idx_scatter, expert_out_scatter, reduce='sum', include_self=False)
            except TypeError: # Fallback for older PyTorch that might not have include_self
                 expert_cache.scatter_reduce_(0, exp_token_idx_scatter, expert_out_scatter, reduce='sum')


        return expert_cache.to(target_device) # Ensure final cache is on target
