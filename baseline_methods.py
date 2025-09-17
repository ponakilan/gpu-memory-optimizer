import numpy as np
from typing import List
from checkpoint_base import CheckpointBase

class BaselineMethods(CheckpointBase):
    def sqrt_n_method(self) -> List[int]:
        n = self.n
        if n <= 1:
            return [0, n]
        
        num_segments = max(1, int(np.sqrt(n)))
        segment_size = n // num_segments
        
        checkpoints = [0]
        for i in range(1, num_segments):
            checkpoints.append(i * segment_size)
        checkpoints.append(n)
        
        return sorted(list(set(checkpoints)))