import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):
	def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
		super().__init__()
		self.model_type = "transformer"
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		encoder_layers = TransformerEncoderLayer(d_model, nlayers)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.encoder = nn.Embedding(ntoken, d_model)
		self.d_model = d_model
		self.decoder = nn.Linear(d_model, ntoken)
		
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src, src_mask):
		"""
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
		src = self.encoder(src) * math.sqrt(self.d_model)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, src_mask)
		output = self.decoder(output)
		return output


def generate_square_subsequent_mask(sz):
	"""Generates an upper-triangular matrix of -inf, with zeros on diag."""
	return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position*div_term)
		pe[:, 0, 1::2] = torch.cos(position*div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)