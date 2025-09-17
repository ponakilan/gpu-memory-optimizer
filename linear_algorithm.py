import numpy as np
from collections import deque
from typing import List, Tuple
from checkpoint_base import CheckpointBase

class LinearCheckpointSelector(CheckpointBase):
    def dynamic_checkpoint_selection_linear(self) -> Tuple[int, List[int]]:
        n = self.n
        d = self.layer_sizes
        
        if n == 0:
            return d[0], [0]
        
        def compute_s(h: int, i: int) -> int:
            if h >= i:
                return 0
            segment_sum = sum(d[h+1:i])
            max_val = max(d[h:i]) if h < i else 0
            return segment_sum + max_val
        
        M = [0] * (n + 1)
        parent = [-1] * (n + 1)
        
        M[n] = d[n]
        
        Q = deque([n])
        j_star = n
        
        for i in range(n - 1, 0, -1):
            j = j_star
            
            min_cost = float('inf')
            best_j = j
            
            for curr_j in range(i + 1, n + 1):
                if curr_j > len(Q) + i:
                    break
                    
                s_val = compute_s(i, curr_j)
                U_val = s_val + d[curr_j]
                
                case1 = d[i] + M[curr_j]
                case2 = d[i] + U_val
                
                cost = max(case1, case2)
                if cost < min_cost:
                    min_cost = cost
                    best_j = curr_j
            
            M[i] = min_cost
            parent[i] = best_j
            j_star = best_j
            
            while Q and M[Q[0]] >= M[i]:
                Q.popleft()
            Q.appendleft(i)
        
        checkpoints = [0]
        curr = 1
        while curr <= n:
            if curr == n or parent[curr] != curr + 1:
                checkpoints.append(curr)
            if parent[curr] != -1:
                curr = parent[curr]
            else:
                break
        
        if n not in checkpoints:
            checkpoints.append(n)
        
        return int(M[1]), sorted(checkpoints)