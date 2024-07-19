import torch
from torch import nn
from math import log


class PositionEmbedding(nn.Module):

    def __init__(self, embed_size, max_len=512):
        super(PositionEmbedding, self).__init__()

        # same size with input matrix
        self.encoding = torch.zeros(max_len, embed_size).float()
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        div_term = torch.arange(0, embed_size, 2).float()
        div_term = (div_term * -(log(10000.0) / embed_size)).exp()

        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)

        self.encoding = nn.Parameter(self.encoding.unsqueeze(0))
        # self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, :x.size(1)]
