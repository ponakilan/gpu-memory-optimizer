from typing import List, Dict
from cubic_algorithm import CubicCheckpointSelector
from linear_algorithm import LinearCheckpointSelector
from baseline_methods import BaselineMethods

class CheckpointOptimizer(CubicCheckpointSelector, LinearCheckpointSelector, BaselineMethods):
    def compare_methods(self) -> Dict:
        results = {}
        
        sqrt_n_checkpoints = self.sqrt_n_method()
        results['sqrt_n'] = {
            'checkpoints': sqrt_n_checkpoints,
            'cost_basic': self.evaluate_checkpoint_cost(sqrt_n_checkpoints, "basic"),
            'cost_pytorch': self.evaluate_checkpoint_cost(sqrt_n_checkpoints, "pytorch")
        }
        
        cost_cubic, checkpoints_cubic = self.checkpoint_selection_cubic()
        results['cubic'] = {
            'checkpoints': checkpoints_cubic,
            'cost_basic': cost_cubic,
            'cost_pytorch': self.evaluate_checkpoint_cost(checkpoints_cubic, "pytorch")
        }
        
        cost_linear, checkpoints_linear = self.dynamic_checkpoint_selection_linear()
        results['linear'] = {
            'checkpoints': checkpoints_linear,
            'cost_pytorch': cost_linear,
            'cost_basic': self.evaluate_checkpoint_cost(checkpoints_linear, "basic")
        }
        
        no_checkpoints = [0, self.n]
        results['none'] = {
            'checkpoints': no_checkpoints,
            'cost_basic': sum(self.layer_sizes),
            'cost_pytorch': sum(self.layer_sizes)
        }
        
        return results