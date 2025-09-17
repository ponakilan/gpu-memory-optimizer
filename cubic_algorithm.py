import numpy as np
from typing import List, Tuple
from checkpoint_base import CheckpointBase

class CubicCheckpointSelector(CheckpointBase):
    def checkpoint_selection_cubic(self) -> Tuple[int, List[int]]:
        n = self.n
        d = self.layer_sizes
        
        segment_costs = set()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                segment_costs.add(sum(d[i+1:j+1]))
        
        segment_costs = sorted(segment_costs)
        min_total_cost = float('inf')
        best_checkpoints = []
        
        for s in segment_costs:
            M = [float('inf')] * (n + 1)
            M[0] = d[0]
            parent = [-1] * (n + 1)
            
            for i in range(1, n + 1):
                l_i = i
                current_sum = 0
                for j in range(i, 0, -1):
                    current_sum += d[j]
                    if current_sum <= s:
                        l_i = j
                    else:
                        break
                
                for j in range(l_i, i + 1):
                    cost = d[j] + M[j - 1] if j > 0 else d[j]
                    if cost < M[i]:
                        M[i] = cost
                        parent[i] = j
            
            total_cost = s + M[n]
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                checkpoints = [0, n]
                curr = n
                while parent[curr] != -1 and parent[curr] != 0:
                    checkpoints.append(parent[curr])
                    curr = parent[curr] - 1
                best_checkpoints = sorted(list(set(checkpoints)))
        
        return int(min_total_cost), best_checkpoints