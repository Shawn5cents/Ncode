"""DeepSeekMoE implementation for Ncode

Implements the Mixture of Experts architecture from DeepSeek-V3
with finer-grained experts and auxiliary-loss-free load balancing.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class DeepSeekMoE(nn.Module):
    """DeepSeek-V3 inspired Mixture of Experts implementation"""
    
    def __init__(self, 
                 num_experts: int = 8,
                 expert_capacity: int = 64,
                 hidden_size: int = 4096,
                 shared_expert_ratio: float = 0.25,
                 num_sub_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.hidden_size = hidden_size
        self.num_sub_experts = num_sub_experts
        
        # Shared expert with sub-experts
        self.shared_expert = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, int(hidden_size * shared_expert_ratio)),
                nn.GELU(),
                nn.Linear(int(hidden_size * shared_expert_ratio), hidden_size)
            )
            for _ in range(num_sub_experts)
        ])
        self.shared_expert_router = nn.Linear(hidden_size, num_sub_experts)
        
        # Routed experts with sub-experts
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                for _ in range(num_sub_experts)
            ])
            for _ in range(num_experts)
        ])
        
        # Routing layers
        self.router = nn.Linear(hidden_size, num_experts)
        self.sub_expert_router = nn.Linear(hidden_size, num_sub_experts)
        
        # Dynamic bias for load balancing
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('sub_expert_bias', torch.zeros(num_sub_experts))
        
        # Complementary sequence-wise auxiliary loss
        self.aux_loss_weight = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer"""
        batch_size = x.size(0)
        
        # Shared expert processing with sub-experts
        shared_router_logits = self.shared_expert_router(x)
        shared_router_probs = F.sigmoid(shared_router_logits + self.sub_expert_bias)
        shared_weights, shared_indices = shared_router_probs.topk(1, dim=-1)
        
        shared_outputs = []
        for i in range(batch_size):
            sub_expert_idx = shared_indices[i]
            expert_out = self.shared_expert[sub_expert_idx](x[i].unsqueeze(0))
            shared_outputs.append(shared_weights[i] * expert_out)
        shared_output = torch.cat(shared_outputs, dim=0)
        
        # Expert routing with sub-experts
        router_logits = self.router(x)
        router_probs = F.sigmoid(router_logits + self.expert_bias)
        
        # Select top-k experts
        top_k = min(2, self.num_experts)
        expert_weights, expert_indices = router_probs.topk(top_k, dim=-1)
        
        # Process through selected experts and their sub-experts
        expert_outputs = []
        for i in range(top_k):
            expert_idx = expert_indices[:, i]
            expert_weight = expert_weights[:, i].unsqueeze(-1)
            
            # Route to sub-experts
            sub_router_logits = self.sub_expert_router(x)
            sub_router_probs = F.sigmoid(sub_router_logits + self.sub_expert_bias)
            sub_weights, sub_indices = sub_router_probs.topk(1, dim=-1)
            
            # Process through selected sub-expert
            expert_out = torch.zeros_like(x)
            for j in range(batch_size):
                sub_expert_idx = sub_indices[j]
                expert_out[j] = self.experts[expert_idx[j]][sub_expert_idx](x[j].unsqueeze(0))
            
            expert_outputs.append(expert_weight * expert_out)
        
        # Combine expert outputs
        routed_output = sum(expert_outputs)
        
        # Calculate complementary sequence-wise auxiliary loss
        self.aux_loss = self.calculate_auxiliary_loss(
            router_probs, shared_router_probs, expert_indices, shared_indices
        )
        
        # Combine shared and routed outputs
        return shared_output + routed_output
    
    def calculate_auxiliary_loss(self, 
                               router_probs: torch.Tensor,
                               shared_router_probs: torch.Tensor,
                               expert_indices: torch.Tensor,
                               shared_indices: torch.Tensor) -> torch.Tensor:
        """Calculate complementary sequence-wise auxiliary loss"""
        # Calculate expert diversity loss
        expert_diversity = router_probs.mean(dim=0)
        expert_diversity_loss = -torch.sum(expert_diversity * torch.log(expert_diversity + 1e-9))
        
        # Calculate shared expert diversity loss
        shared_diversity = shared_router_probs.mean(dim=0)
        shared_diversity_loss = -torch.sum(shared_diversity * torch.log(shared_diversity + 1e-9))
        
        # Calculate utilization balance loss
        expert_util = self.get_expert_utilization(expert_indices)
        shared_util = self.get_sub_expert_utilization(shared_indices)
        util_balance_loss = torch.sum((expert_util - 1/self.num_experts)**2) + \
                          torch.sum((shared_util - 1/self.num_sub_experts)**2)
        
        return self.aux_loss_weight * (expert_diversity_loss + shared_diversity_loss + util_balance_loss)

    def update_expert_bias(self, expert_utilization: torch.Tensor):
        """Update dynamic bias for load balancing"""
        # Calculate new bias based on expert utilization
        target_utilization = 1.0 / self.num_experts
        utilization_diff = expert_utilization - target_utilization
        self.expert_bias -= 0.1 * utilization_diff  # Adjust expert bias
        
        # Update sub-expert bias based on shared expert utilization
        shared_utilization = self.get_sub_expert_utilization()
        target_shared_utilization = 1.0 / self.num_sub_experts
        shared_utilization_diff = shared_utilization - target_shared_utilization
        self.sub_expert_bias -= 0.1 * shared_utilization_diff  # Adjust sub-expert bias
        
    def get_expert_utilization(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Calculate expert utilization statistics"""
        # Create one-hot encoding of expert indices
        one_hot = F.one_hot(expert_indices, num_classes=self.num_experts)
        # Calculate utilization as mean over batch
        return one_hot.float().mean(dim=0)
        
    def get_sub_expert_utilization(self, sub_expert_indices: torch.Tensor = None) -> torch.Tensor:
        """Calculate sub-expert utilization statistics"""
        if sub_expert_indices is None:
            # If no indices provided, return uniform distribution
            return torch.ones(self.num_sub_experts) / self.num_sub_experts
            
        # Create one-hot encoding of sub-expert indices
        one_hot = F.one_hot(sub_expert_indices, num_classes=self.num_sub_experts)
        # Calculate utilization as mean over batch
        return one_hot.float().mean(dim=0)
