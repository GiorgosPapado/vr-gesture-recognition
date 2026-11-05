import torch
import torch.nn as nn
from torch.nn.functional import interpolate

class TSInterpolator(nn.Module):

    def __init__(self, length: int):
        """
        :param length: the target interpolation length
        """
        super().__init__()
        self.length = length


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        :param x: timeseries input of size BxFxT (with B denoting batch size, T time series length and F the total number of features)
        :return: BxFxL tensor with all F time-series of length T interpolated to length L
        """

        return interpolate(x, size=self.length, mode='linear', align_corners=True)