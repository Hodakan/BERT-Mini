from torch import nn
from math import sqrt


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # get matrix q, k, v
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # do scale dot product to caculate similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / sqrt(self.d_model // self.n_head)

        # apply masking
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = nn.functional.softmax(score)
        out = score @ v

        # concat and pass to linear layer
        out = self.concat(out)
        out = self.fc_out(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of heads
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)
        tensor = tensor.transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split()
        """
        batch_size, head, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(
            batch_size, length, self.d_model)
        return tensor
