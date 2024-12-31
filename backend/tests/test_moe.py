"""Tests for Mixture of Experts implementation."""

import unittest
import torch
from backend.experimental.deepseek_moe import DeepSeekMoE

class TestDeepSeekMoE(unittest.TestCase):
    def setUp(self):
        self.num_experts = 8
        self.expert_capacity = 64
        self.hidden_size = 4096
        self.shared_expert_ratio = 0.25
        self.num_sub_experts = 4
        self.moe = DeepSeekMoE(
            num_experts=self.num_experts,
            expert_capacity=self.expert_capacity,
            hidden_size=self.hidden_size,
            shared_expert_ratio=self.shared_expert_ratio,
            num_sub_experts=self.num_sub_experts
        )

    def test_initialization(self):
        self.assertEqual(self.moe.num_experts, self.num_experts)
        self.assertEqual(self.moe.expert_capacity, self.expert_capacity)
        self.assertEqual(self.moe.hidden_size, self.hidden_size)
        self.assertEqual(self.moe.shared_expert_ratio, self.shared_expert_ratio)
        
        # Verify shared expert dimensions with sub-experts
        self.assertEqual(len(self.moe.shared_expert), self.num_sub_experts)
        for sub_expert in self.moe.shared_expert:
            self.assertEqual(len(sub_expert), 3)  # Linear -> GELU -> Linear
            self.assertEqual(sub_expert[0].in_features, self.hidden_size)
            self.assertEqual(sub_expert[0].out_features, int(self.hidden_size * self.shared_expert_ratio))
            self.assertEqual(sub_expert[2].in_features, int(self.hidden_size * self.shared_expert_ratio))
            self.assertEqual(sub_expert[2].out_features, self.hidden_size)
        
        # Verify shared expert router
        self.assertEqual(self.moe.shared_expert_router.in_features, self.hidden_size)
        self.assertEqual(self.moe.shared_expert_router.out_features, self.num_sub_experts)
        
        # Verify routed experts with sub-experts
        self.assertEqual(len(self.moe.experts), self.num_experts)
        for expert in self.moe.experts:
            self.assertEqual(len(expert), self.num_sub_experts)
            for sub_expert in expert:
                self.assertEqual(len(sub_expert), 3)  # Linear -> GELU -> Linear
                self.assertEqual(sub_expert[0].in_features, self.hidden_size)
                self.assertEqual(sub_expert[0].out_features, self.hidden_size)
                self.assertEqual(sub_expert[2].in_features, self.hidden_size)
                self.assertEqual(sub_expert[2].out_features, self.hidden_size)
        
        # Verify routers
        self.assertEqual(self.moe.router.in_features, self.hidden_size)
        self.assertEqual(self.moe.router.out_features, self.num_experts)
        self.assertEqual(self.moe.sub_expert_router.in_features, self.hidden_size)
        self.assertEqual(self.moe.sub_expert_router.out_features, self.num_sub_experts)

    def test_forward_pass(self):
        batch_size = 16
        x = torch.randn(batch_size, self.hidden_size)
        
        output = self.moe(x)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_size, self.hidden_size))
        
        # Verify output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

    def test_expert_utilization(self):
        batch_size = 16
        x = torch.randn(batch_size, self.hidden_size)
        
        # Get router probabilities
        router_logits = self.moe.router(x)
        router_probs = torch.sigmoid(router_logits + self.moe.expert_bias)
        
        # Select top-k experts
        top_k = min(2, self.num_experts)
        expert_weights, expert_indices = router_probs.topk(top_k, dim=-1)
        
        # Calculate utilization
        utilization = self.moe.get_expert_utilization(expert_indices)
        
        # Verify utilization shape and values
        self.assertEqual(utilization.shape, (self.num_experts,))
        self.assertTrue(torch.all(utilization >= 0))
        self.assertTrue(torch.all(utilization <= 1))
        self.assertAlmostEqual(utilization.sum().item(), 1.0, places=5)

    def test_bias_update(self):
        initial_bias = self.moe.expert_bias.clone()
        initial_sub_bias = self.moe.sub_expert_bias.clone()
        
        # Create fake utilization
        utilization = torch.ones(self.num_experts) / self.num_experts
        utilization[0] = 0.5  # Make first expert overutilized
        
        # Update bias
        self.moe.update_expert_bias(utilization)
        
        # Verify expert bias changed
        self.assertFalse(torch.allclose(self.moe.expert_bias, initial_bias))
        self.assertLess(self.moe.expert_bias[0], initial_bias[0])
        
        # Verify sub-expert bias changed
        self.assertFalse(torch.allclose(self.moe.sub_expert_bias, initial_sub_bias))

    def test_auxiliary_loss(self):
        batch_size = 16
        x = torch.randn(batch_size, self.hidden_size)
        
        # Forward pass to calculate loss
        _ = self.moe(x)
        
        # Verify auxiliary loss exists and is a scalar
        self.assertTrue(hasattr(self.moe, 'aux_loss'))
        self.assertEqual(self.moe.aux_loss.dim(), 0)
        
        # Verify loss is non-negative
        self.assertTrue(self.moe.aux_loss >= 0)

    def test_sub_expert_utilization(self):
        batch_size = 16
        x = torch.randn(batch_size, self.hidden_size)
        
        # Get sub-expert indices
        router_logits = self.moe.sub_expert_router(x)
        router_probs = torch.sigmoid(router_logits + self.moe.sub_expert_bias)
        _, sub_expert_indices = router_probs.topk(1, dim=-1)
        
        # Calculate utilization
        utilization = self.moe.get_sub_expert_utilization(sub_expert_indices)
        
        # Verify utilization shape and values
        self.assertEqual(utilization.shape, (self.num_sub_experts,))
        self.assertTrue(torch.all(utilization >= 0))
        self.assertTrue(torch.all(utilization <= 1))
        self.assertAlmostEqual(utilization.sum().item(), 1.0, places=5)

    def test_shared_expert_routing(self):
        batch_size = 16
        x = torch.randn(batch_size, self.hidden_size)
        
        # Get shared expert routing
        router_logits = self.moe.shared_expert_router(x)
        router_probs = torch.sigmoid(router_logits + self.moe.sub_expert_bias)
        weights, indices = router_probs.topk(1, dim=-1)
        
        # Verify routing results
        self.assertEqual(weights.shape, (batch_size, 1))
        self.assertEqual(indices.shape, (batch_size, 1))
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < self.num_sub_experts))

if __name__ == '__main__':
    unittest.main()
