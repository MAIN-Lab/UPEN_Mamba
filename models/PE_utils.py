import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveExpansionQKV(nn.Module):
    def __init__(self, in_channels, out_channels, expand_type=1, function='arctan'):
        super().__init__()
        self.expand_type = expand_type
        self.function = function

        # 1×1 convolution layers for Q, K, V
        # They will each produce out_channels = `out_channels`
        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Polynomial expansions
        if self.expand_type == 1:
            # Approximate expansions for arctan(x) up to x^3
            p1 = x
            p2 = x - (x ** 2) / 2
            p3 = x - (x ** 2) / 2 + (x ** 3) / 3
        else:
            # Example alternative expansions
            p1 = x
            p2 = -(x ** 2) / 2
            p3 = (x ** 3) / 3

        # Project each expanded map to Q, K, V
        q = self.conv_q(p1)
        k = self.conv_k(p2)
        v = self.conv_v(p3)

        return q, k, v
