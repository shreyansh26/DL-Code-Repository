import torch
from torch import nn
import torch.nn.functional as F

from tfb.modules import TransfomerBlock
from .utils import device_


class GTransformer(nn.Module):
    """
    Transformer for generating text (char by char)
    """
    def __init__(self, emb, heads, depth, seq_length, num_tokens):
        """
        Params:
            emb: Embedding dimension
            heads: nr. of attention heads
            depth: Number of transformer blocks
            seq_length: Expected maximum sequence length
            num_tokens: Number of tokens (usually words) in the vocabulary
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        transformer_blocks = []

        for i in range(depth):
            transformer_blocks.append(TransfomerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, pos_embedding=self.pos_embedding))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        Params:
            x: A (batch, sequence length) integer tensor of token indices
        Returns:
            Predicted log-probability vectors for each token based on the preceding tokens
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=device_()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.transformer_blocks(x)

        x = self.to_probs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)


class CTransformer(nn.Module):
    """
    Transformer for classifying sequences 
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0):
        """
        Params:
            emb: Embedding dimension
            heads: nr. of attention heads
            depth: Number of transformer blocks
            seq_length: Expected maximum sequence length
            num_tokens: Number of tokens (usually words) in the vocabulary
            num_classes: Number of classes.
            max_pool: If true, use global max pooling in the last layer. If false, use global average pooling.
            dropout: Percentage dropout
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.max_pool = max_pool
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        transformer_blocks = []

        for i in range(depth):
            transformer_blocks.append(TransfomerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probs = nn.Linear(emb, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Params:
            x: A batch by sequence length integer tensor of token indices
        Retruns:
            Predicted log-probability vectors for each token based on the preceding tokens
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=device_()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.dropout(x)

        x = self.transformer_blocks(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.to_probs(x)

        return F.log_softmax(x, dim=1) # Since not row-wise, but class wise softmax required here.

