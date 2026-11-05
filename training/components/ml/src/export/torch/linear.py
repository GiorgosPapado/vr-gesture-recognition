import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, weights : torch.Tensor, intercept: torch.Tensor):
        # weights       size: class_count X n_features
        # intercept     size: class_count
        super().__init__()
        assert len(weights) > 2, "Class Count should be > 2. This porting of RidgeClassifierCV does not support binary classification"
        self.register_buffer('weights', weights.float())
        self.register_buffer('intercept', intercept.float())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: BxF Batch x Features
        :returns: BxC logits Batch x Class count
        """

        return x @ self.weights.transpose(1,0) + self.intercept