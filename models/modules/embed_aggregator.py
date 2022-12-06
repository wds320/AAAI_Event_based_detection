import torch
import torch.nn as nn
from apex import amp


class EmbedAggregator(nn.Module):
    """
    Aggregate feature maps of neighboring time interval.
    """
    def __init__(self, channels, kernel_size=3):
        super(EmbedAggregator, self).__init__()
        self.embed_convs = nn.Sequential(nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                                         nn.ReLU(inplace=True))

    def forward(self, curr_x, prev_x):
        """
        1.Compute the cos similarity between current feature map and previous feature map. e.g. Ft, Ft-1
        2.Use the normalized(softmax)cos similarity to weightedly hidden state
        Args:
            curr_x: [1, C, H, W]
            prev_x: [1, C, H, W]
        Returns:
            weights: [1, 1, H, W]
        """
        curr_embed = self.embed_convs(curr_x)
        prev_embed = self.embed_convs(prev_x)

        curr_embed = curr_embed / (curr_embed.norm(p=2, dim=1, keepdim=True) + 1e-6)  # L2
        prev_embed = prev_embed / (prev_embed.norm(p=2, dim=1, keepdim=True) + 1e-6)

        weights = torch.sum(curr_embed*prev_embed, dim=1, keepdim=True)
        return weights
