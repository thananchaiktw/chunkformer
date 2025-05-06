"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F

class StreamingRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 50000) -> None:
        """Construct an PositionalEncoding object."""
        super(StreamingRelPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.xscale = math.sqrt(self.d_model)
        self.max_len = max_len
        self.extend_pe(max_len)

    def extend_pe(self, size: int, left_context: Union[int, torch.Tensor] = 0) -> None:
        """Reset the positional encodings."""
        x_size_1 = size + left_context

        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size_1, self.d_model)
        pe_negative = torch.zeros(x_size_1, self.d_model)
        position = torch.arange(0, x_size_1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int,
                          apply_dropout: bool = False, 
                          right_context_size: Union[int, torch.Tensor] = 0) -> torch.Tensor:

        if isinstance(offset, int):
            assert offset + size < self.max_len
            x_size_1 = size + offset
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + size + right_context_size,
            ]
        else:
            assert offset + size < self.max_len
            x_size_1 = size + offset
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + size + right_context_size,
            ]

        return pos_emb

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0,
        right_context_size: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            offset (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False, right_context_size).to(device=x.device, dtype=x.dtype)
        return self.dropout(x), self.dropout(pos_emb)

class StreamingRotaryPositionalEncoding(torch.nn.Module):
    """
    Streaming rotary positional encoding (RoFormer style) for chunked/streaming inputs.

    Precomputes sin and cos caches up to max_seq_len, and applies rotation
    based on a position offset (accumulated frames) for each chunk.

    Args:
        d_model: Embedding dimension (must be even).
        dropout_rate: Dropout on the output.
        max_seq_len: Maximum sequence length to precompute.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.0, max_seq_len: int = 50000) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies and precompute sin/cos caches
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))  # (d_model/2,)
        pos_seq = torch.arange(0, max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        # Outer to get frequencies for each position
        freqs = torch.einsum('i,j->ij', pos_seq, inv_freq)  # (max_seq_len, d_model/2)
        # Build sin and cos caches
        sin_cache = freqs.sin()  # (max_seq_len, d_model/2)
        cos_cache = freqs.cos()
        # Register as buffers so they move with the module
        self.register_buffer('sin_cache', sin_cache, persistent=False)
        self.register_buffer('cos_cache', cos_cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Apply rotary positional encoding to a chunk.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).
            offset: Starting position index for this chunk (int).

        Returns:
            Tensor of same shape with rotary pos-emb applied.
        """
        bsz, seq_len, dim = x.size()
        device = x.device
        # Slice sin/cos for this chunk
        sin = self.sin_cache[offset: offset + seq_len].to(device)  # (seq_len, d_model/2)
        cos = self.cos_cache[offset: offset + seq_len].to(device)

        # Expand to (batch, seq_len, d_model/2)
        sin = sin.unsqueeze(0).expand(bsz, -1, -1)
        cos = cos.unsqueeze(0).expand(bsz, -1, -1)

        # Apply rotation: split last dim
        x1, x2 = x[..., :dim//2], x[..., dim//2:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return self.dropout(x_rotated)
