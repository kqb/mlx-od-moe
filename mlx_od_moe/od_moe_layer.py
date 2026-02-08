"""
OD-MoE Layer - Core on-demand expert loading layer

Only loads top-k experts into working memory regardless of total expert count (384).
Triggers prefetch for next layer based on shadow model predictions.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, List, Dict
from .expert_store import UnifiedMemoryExpertStore
from .shadow_model import ShadowRunner


class ODMoELayer(nn.Module):
    """
    On-Demand MoE layer.
    
    Key feature: Only loads 8 experts into working memory regardless
    of total expert count (384 for Kimi-K2.5).
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_dim: int = 4096,
        ffn_dim: int = 14336,
        num_experts: int = 384,
        top_k: int = 8,
        expert_store: Optional[UnifiedMemoryExpertStore] = None,
        shadow_runner: Optional[ShadowRunner] = None,
        n_shared_experts: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        num_layers: int = 28,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_layers = num_layers

        self.expert_store = expert_store
        self.shadow_runner = shadow_runner

        # Router/gate (always resident, small: 4096 * 384 ≈ 1.5MB)
        self.gate = nn.Linear(hidden_dim, num_experts)

        # Shared expert (always resident, runs on every token)
        self.n_shared_experts = n_shared_experts
        if n_shared_experts > 0:
            shared_dim = n_shared_experts * ffn_dim
            self.shared_gate_proj = nn.Linear(hidden_dim, shared_dim, bias=False)
            self.shared_up_proj = nn.Linear(hidden_dim, shared_dim, bias=False)
            self.shared_down_proj = nn.Linear(shared_dim, hidden_dim, bias=False)
        else:
            self.shared_gate_proj = None

        # Currently loaded experts (working set)
        # Format: {expert_idx: (w1, w2, w3)}
        self.active_experts = {}

        # Expert architecture: Gated MLP
        # w1: up-projection (hidden → ffn)
        # w2: down-projection (ffn → hidden)
        # w3: gate-projection (hidden → ffn)
        self.expert_w1_dim = ffn_dim
        self.expert_w2_dim = hidden_dim
        self.expert_w3_dim = ffn_dim

        # Telemetry for expert usage
        self.expert_usage_counts = {}  # {expert_idx: count}
        self.total_expert_selections = 0
        self.aux_loss = None  # Load balancing auxiliary loss
    
    def load_experts(self, expert_indices: List[int]):
        """
        Load specific experts into working memory from store.
        
        Non-blocking if already cached in expert store.
        
        Args:
            expert_indices: List of expert IDs to load
        """
        if not self.expert_store:
            # Fallback: no expert store, use dummy weights
            return
        
        new_experts = {}
        for idx in expert_indices:
            # Skip if already in active set
            if idx in self.active_experts:
                continue
            
            # Fetch from expert store (fast if cached, slow if SSD read)
            expert_weights = self.expert_store.fetch(self.layer_idx, idx)

            if isinstance(expert_weights, dict):
                # Dict format from real expert store (safetensors files)
                w1 = expert_weights['w1']
                w2 = expert_weights['w2']
                w3 = expert_weights['w3']
            else:
                # Flat array format (for testing)
                w1_size = self.hidden_dim * self.expert_w1_dim
                w2_size = self.expert_w1_dim * self.expert_w2_dim
                w3_size = self.hidden_dim * self.expert_w3_dim

                w1_flat = expert_weights[:w1_size]
                w2_flat = expert_weights[w1_size:w1_size + w2_size]
                w3_flat = expert_weights[w1_size + w2_size:w1_size + w2_size + w3_size]

                w1 = w1_flat.reshape(self.hidden_dim, self.expert_w1_dim)
                w2 = w2_flat.reshape(self.expert_w1_dim, self.expert_w2_dim)
                w3 = w3_flat.reshape(self.hidden_dim, self.expert_w3_dim)

            new_experts[idx] = (w1, w2, w3)
        
        # Update active set
        self.active_experts.update(new_experts)
        
        # Cleanup: remove experts not in current top-k
        current_top_k = set(expert_indices)
        for idx in list(self.active_experts.keys()):
            if idx not in current_top_k:
                del self.active_experts[idx]
    
    def _compute_load_balancing_loss(
        self,
        router_probs: mx.array,
        top_k_indices: mx.array
    ) -> float:
        """
        Compute auxiliary load balancing loss.

        Encourages uniform expert usage across the batch.
        Lower values indicate better load distribution.

        Args:
            router_probs: Router probabilities (num_tokens, num_experts)
            top_k_indices: Selected expert indices (num_tokens, top_k)

        Returns:
            Scalar load balancing loss
        """
        num_tokens = router_probs.shape[0]

        # Create mask for selected experts
        expert_mask = mx.zeros((num_tokens, self.num_experts))
        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                expert_mask[i, expert_idx] = 1.0

        # Expert usage frequency (fraction of tokens routing to each expert)
        expert_usage = mx.mean(expert_mask, axis=0)  # (num_experts,)

        # Router probability mass assigned to each expert
        expert_prob_mass = mx.mean(router_probs, axis=0)  # (num_experts,)

        # Load balancing loss: dot product encourages balance
        # Scale by num_experts to normalize
        return (mx.sum(expert_usage * expert_prob_mass) * self.num_experts).item()

    def apply_expert(self, x: mx.array, expert_idx: int) -> mx.array:
        """
        Apply single expert: SwiGLU FFN.
        
        Formula: y = (silu(x @ w1) * (x @ w3)) @ w2
        
        Args:
            x: Input tensor (batch*seq, hidden_dim)
            expert_idx: Which expert to apply
        
        Returns:
            Output tensor (batch*seq, hidden_dim)
        """
        if expert_idx not in self.active_experts:
            # Expert not loaded, return zeros (shouldn't happen)
            return mx.zeros_like(x)
        
        w1, w2, w3 = self.active_experts[expert_idx]

        # SwiGLU: (silu(x @ w1) * (x @ w3)) @ w2
        # where silu(x) = x * sigmoid(x)
        gate_proj = x @ w1
        up_proj = x @ w3

        # SiLU activation
        silu_gate = gate_proj * mx.sigmoid(gate_proj)

        # Element-wise product (gating)
        intermediate = silu_gate * up_proj

        # Down-projection
        output = intermediate @ w2

        return output
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with on-demand expert loading.

        Steps:
        1. Compute router logits
        2. Select top-k experts per token
        3. Fetch those experts (cold: SSD, hot: cache)
        4. Compute weighted sum of expert outputs
        5. Prefetch for next layer (async)

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)

        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        batch, seq_len, hidden = x.shape

        # Flatten batch/sequence for routing
        x_flat = x.reshape(-1, hidden)  # (batch*seq, hidden)
        num_tokens = x_flat.shape[0]

        # Router logits
        router_logits = self.gate(x_flat)  # (batch*seq, num_experts)

        # Scoring function: sigmoid for K2.5, softmax for others
        if self.scoring_func == "sigmoid":
            router_probs = mx.sigmoid(router_logits)
        else:
            router_probs = mx.softmax(router_logits, axis=-1)

        # Select top-k experts per token
        top_k_indices = mx.argsort(router_probs, axis=-1)[:, -self.top_k:]

        # Gather top-k weights for selected experts
        # Shape: (num_tokens, top_k)
        top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)

        # Apply routing scaling factor (K2.5 uses 2.827)
        top_k_weights = top_k_weights * self.routed_scaling_factor

        # Compute load balancing auxiliary loss
        self.aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)

        # Determine unique experts needed for this batch
        unique_experts = np.unique(top_k_indices.tolist()).tolist()

        # Load working set (this is the "OD" in OD-MoE)
        self.load_experts(unique_experts)

        # Track expert usage for telemetry
        for expert_idx in unique_experts:
            self.expert_usage_counts[expert_idx] = \
                self.expert_usage_counts.get(expert_idx, 0) + 1
        self.total_expert_selections += len(unique_experts)

        # Optimized batched expert application using broadcasting
        output = mx.zeros_like(x_flat)

        # Process all experts in parallel using vectorized operations
        for expert_idx in unique_experts:
            if expert_idx not in self.active_experts:
                continue

            # Create mask for where this expert is selected (num_tokens, top_k)
            expert_selected = (top_k_indices == expert_idx)

            # Get weights for this expert where it's selected
            # Shape: (num_tokens, top_k) -> broadcast to (num_tokens,)
            expert_weights_masked = mx.where(
                expert_selected,
                top_k_weights,
                mx.zeros_like(top_k_weights)
            )
            # Sum across top_k to get per-token weight for this expert
            expert_weights_per_token = mx.sum(expert_weights_masked, axis=-1)  # (num_tokens,)

            # Apply expert to all tokens (even if weight is 0, will be masked out)
            expert_output = self.apply_expert(x_flat, expert_idx)  # (num_tokens, hidden)

            # Weighted addition: expert_weights_per_token is (num_tokens,), broadcast to (num_tokens, 1)
            output = output + expert_output * expert_weights_per_token[:, None]

        # Add shared expert output (always-on, not routed)
        if self.shared_gate_proj is not None:
            gate = self.shared_gate_proj(x_flat)
            shared_out = self.shared_down_proj(
                gate * mx.sigmoid(gate) * self.shared_up_proj(x_flat)
            )
            output = output + shared_out

        # Trigger prefetch for next layer
        if self.shadow_runner and self.layer_idx < self.num_layers - 1:
            next_experts = self.shadow_runner.get_predictions_for_layer(self.layer_idx + 1)
            if self.expert_store:
                self.expert_store.prefetch(self.layer_idx + 1, next_experts)

        # Reshape back
        return output.reshape(batch, seq_len, hidden)

    def get_expert_usage_stats(self) -> Dict:
        """
        Get expert usage statistics for monitoring load balance.

        Returns:
            Dictionary with:
            - expert_counts: dict mapping expert_idx to usage count
            - total_selections: total number of expert selections
            - load_balance_coefficient: measure of imbalance (1.0 = perfect balance)
        """
        if self.total_expert_selections == 0:
            return {
                'expert_counts': {},
                'total_selections': 0,
                'load_balance_coefficient': 1.0
            }

        # Compute load balance coefficient
        # Perfect balance: all experts used equally -> coeff = 1.0
        # Higher values indicate imbalance
        counts = list(self.expert_usage_counts.values())
        if not counts:
            load_balance_coeff = 1.0
        else:
            # Coefficient of variation: std / mean
            mean_usage = np.mean(counts)
            std_usage = np.std(counts)
            # Add 1.0 to convert CV to coefficient (1.0 = perfect, >1.0 = imbalanced)
            load_balance_coeff = 1.0 + (std_usage / mean_usage if mean_usage > 0 else 0.0)

        return {
            'expert_counts': dict(self.expert_usage_counts),
            'total_selections': self.total_expert_selections,
            'load_balance_coefficient': float(load_balance_coeff)
        }
