from tfb import utils
from tfb.utils import mask_

import torch
from torch import nn
import torch.nn.functional as F

import random
import math
import sys


class SelfAttention(nn.Module):
    """
    Basic implemetation of multi-head self attention
    """

    def __init__(self, emb, heads=8, mask=False):
        """
        Params:
            emb: Embedding size
            heads: Number of attention heads
            mask: Is masked self attention required?
        """
        super().__init__()

        assert emb % heads == 0, f'Embedding size - {emb} must be divisible by the number of attention heads - {heads}'

        self.emb = emb
        self.heads = heads    # Embedding will be borken into {heads} chunks and feed each into a different attention head
        self.mask = mask


        s = emb // heads

        self.to_keys = nn.Linear(emb, emb, bias=False)
        self.to_queries = nn.Linear(emb, emb, bias=False)
        self.to_values = nn.Linear(emb, emb, bias=False)

        self.unify_heads = nn.Linear(emb, emb)

    def forward(self, x):
        """
        Params:
            x: Input sequence
        """

        b, t, e = x.size()
        h = self.heads

        assert e == self.emb, f'Input embedding size - {e} must be same as layer embedding dim - {self.emb}'

        s = e // h

        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # Now after computing on complete embedding vector, we split them into different heads

        # Scaled dot product self-attention
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Instead of dividing dot product with sqrt(e), divide both query and key by sqrt(sqrt(e)). Memory efficient
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        # Take dot product of queries and keys and scale (Batch wise multiplication)
        dot_p = torch.bmm(queries, keys.transpose(1, 2))

        assert dot_p.size() == (b*h, t, t)

        if self.mask:    # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot_p, maskval=float('-inf'), mask_diagonal=False)

        # Row-wise probabilities
        dot_p = F.softmax(dot_p, dim=2)

        # Self attention
        out = torch.bmm(dot_p, values).view(b, h, t, s)

        # Swap back h and t
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        return self.unify_heads(out)


class TransfomerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, hidden_mult=4, dropout=0.0, pos_embedding=None):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(hidden_mult * emb, emb)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)
        x = self.dropout(x)

        return x