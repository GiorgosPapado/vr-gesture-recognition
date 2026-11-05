import torch
import torch.nn as nn
from typing import List

class Pipeline(nn.Module):

    def __init__(self, steps: List[nn.Module]):
        super().__init__()
        self.steps = nn.ModuleList(steps)

    def forward(self, x : torch.Tensor) -> torch.Tensor:        
        for step in self.steps:
            x = step(x)
        return x