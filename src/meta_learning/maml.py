"""Model-Agnostic Meta-Learning (MAML) implementation from first principles."""
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) engine for few-shot channel estimation.
    
    Implements:
        Inner Loop: theta_i' = theta - alpha * grad_theta L(D_i^S; theta)
        Outer Loop: theta <- theta - beta * grad_theta sum_i L(D_i^Q; theta_i')
        
    Attributes:
        model (nn.Module): Base learner (e.g. ClassicalMLP, ClassicalCNN, or HybridVQC).
        inner_lr (float): Inner-loop adaptation learning rate (alpha).
        outer_lr (float): Outer-loop meta-learning rate (beta).
        first_order (bool): If True, use First-Order MAML (FOMAML); otherwise full second-order MAML.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        first_order: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.device = device
        self.criterion = nn.MSELoss()
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        
    def get_parameter_dict(self) -> OrderedDict:
        """Extract ordered dictionary of named parameters."""
        return OrderedDict((name, param) for name, param in self.model.named_parameters() if param.requires_grad)
        
    def forward_with_params(self, x: torch.Tensor, params: Optional[OrderedDict] = None) -> torch.Tensor:
        """Forward pass using either provided parameters or the model's current parameters."""
        if params is None:
            return self.model(x)
        return torch.func.functional_call(self.model, params, (x,))
        
    def adapt(
        self,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        num_steps: int = 5,
        inner_lr: Optional[float] = None,
        create_graph: bool = False,
    ) -> OrderedDict:
        """Perform inner-loop few-shot gradient adaptation on support pilots.
        
        Args:
            x_support: Support feature tensor (K, 36).
            y_support: Support target tensor (K, 2).
            num_steps: Number of gradient adaptation steps.
            inner_lr: Learning rate alpha (defaults to self.inner_lr).
            create_graph: Whether to retain computation graph for second-order meta-gradients.
            
        Returns:
            Adapted parameter dictionary theta_i'.
        """
        lr = self.inner_lr if inner_lr is None is None else inner_lr
        params = self.get_parameter_dict()
        
        if num_steps == 0:
            return params
            
        adapted_params = OrderedDict()
        for name, param in params.items():
            adapted_params[name] = param
            
        for step in range(num_steps):
            # Evaluate support loss with current adapted parameters
            pred_support = self.forward_with_params(x_support, adapted_params)
            support_loss = self.criterion(pred_support, y_support)
            
            # Compute gradients with respect to adapted parameters
            grads = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=create_graph and (not self.first_order),
                retain_graph=True if step < num_steps - 1 else create_graph,
                allow_unused=True,
            )
            
            # Inner-loop gradient descent update: theta' = theta - alpha * grad
            updated_params = OrderedDict()
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is None:
                    updated_params[name] = param
                else:
                    updated_params[name] = param - lr * grad
            adapted_params = updated_params
            
        return adapted_params
        
    def meta_loss(
        self,
        meta_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]],
        inner_steps: int = 5,
    ) -> torch.Tensor:
        """Compute aggregate meta-loss across query sets of a batch of tasks."""
        total_query_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        batch_size = len(meta_batch)
        if batch_size == 0:
            return total_query_loss
            
        for x_s, y_s, x_q, y_q, _ in meta_batch:
            x_s, y_s = x_s.to(self.device), y_s.to(self.device)
            x_q, y_q = x_q.to(self.device), y_q.to(self.device)
            
            # 1. Inner adaptation on Support set D^S
            adapted_params = self.adapt(x_s, y_s, num_steps=inner_steps, create_graph=True)
            
            # 2. Evaluate adapted model on Query set D^Q
            pred_query = self.forward_with_params(x_q, adapted_params)
            query_loss = self.criterion(pred_query, y_q)
            
            total_query_loss = total_query_loss + query_loss
            
        return total_query_loss / batch_size
        
    def meta_step(
        self,
        meta_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]],
        inner_steps: int = 5,
    ) -> float:
        """Perform one outer-loop meta-optimization step."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        loss = self.meta_loss(meta_batch, inner_steps=inner_steps)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        return loss.item()
        
    def evaluate_adaptation(
        self,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        x_query: torch.Tensor,
        y_query: torch.Tensor,
        adaptation_steps: List[int] = [0, 1, 3, 5, 10],
        inner_lr: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate few-shot adaptation performance across varying numbers of gradient steps."""
        self.model.eval()
        x_s = x_support.to(self.device)
        y_s = y_support.to(self.device)
        x_q = x_query.to(self.device)
        y_q = y_query.to(self.device)
        
        from src.evaluation.metrics import compute_all_metrics
        
        results_by_step = {}
        for steps in adaptation_steps:
            with torch.enable_grad():
                adapted_params = self.adapt(x_s, y_s, num_steps=steps, inner_lr=inner_lr, create_graph=False)
                
            with torch.no_grad():
                pred_q = self.forward_with_params(x_q, adapted_params)
                metrics = compute_all_metrics(y_q.cpu().numpy(), pred_q.cpu().numpy())
                results_by_step[steps] = metrics
                
        return results_by_step
