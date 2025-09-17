import numpy as np
from typing import List, Tuple

class CheckpointBase:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = np.array(layer_sizes)
        self.n = len(layer_sizes) - 1

    def evaluate_checkpoint_cost(self, checkpoints: List[int], algorithm_type: str = "pytorch") -> int:
        d = self.layer_sizes
        checkpoints = sorted(set(checkpoints + [0, self.n]))
        
        if algorithm_type == "basic":
            checkpoint_memory = sum(d[i] for i in checkpoints)
            
            max_segment_memory = 0
            for i in range(len(checkpoints) - 1):
                start = checkpoints[i]
                end = checkpoints[i + 1]
                segment_memory = sum(d[start + 1:end])
                max_segment_memory = max(max_segment_memory, segment_memory)
            
            return checkpoint_memory + max_segment_memory
        
        else:
            max_memory = 0
            
            for i in range(len(checkpoints) - 1, 0, -1):
                curr_checkpoint = checkpoints[i]
                prev_checkpoint = checkpoints[i - 1]
                
                checkpoint_mem = sum(d[j] for j in checkpoints[:i+1])
                segment_mem = sum(d[prev_checkpoint + 1:curr_checkpoint])
                gradient_buffer = max(d[prev_checkpoint:curr_checkpoint]) if prev_checkpoint < curr_checkpoint else 0
                
                total_mem = checkpoint_mem + segment_mem + gradient_buffer
                max_memory = max(max_memory, total_mem)
            
            return max_memory