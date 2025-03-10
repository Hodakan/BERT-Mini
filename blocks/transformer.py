from torch import nn

from layers.layer_norm import LayerNorm
from layers.attention import MultiHeadAttention
from layers.feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # compute attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # position wise feed forward network
        _x = x
        x = self.ffn(x)

        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
