import torch
import torch.nn as nn

class StandardScaler(nn.Module):
    
    def __init__(self, 
                 with_mean: bool,
                 with_std: bool,
                 mean: torch.Tensor = None,
                 scale: torch.Tensor = None):
        
        super().__init__()

        self.with_mean = with_mean
        self.with_std = with_std
        self.mean = mean
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """
        :param x: size BxF (Batch X Features)        
        """
        if self.with_mean:
            x -= self.mean
        if self.with_std:
            x /= self.scale
        
        return x
