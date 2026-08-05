import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, layers):
        super().__init__()
        self.in_map = nn.Linear(input_dim, hid_dim)
        self.out_map = nn.Linear(hid_dim, output_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=hid_dim, out_features=hid_dim) for _ in range(layers)]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.in_map(x)
        for layer in self.layers:
            y = self.activation(layer(y))

        y = self.out_map(y)
        return y


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, d_h, d_v, n_heads):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_h = d_h
        self.d_v = d_v

        self.fc_q = nn.Linear(hid_dim, d_h * n_heads)
        self.fc_k = nn.Linear(hid_dim, d_h * n_heads)
        self.fc_v = nn.Linear(hid_dim, d_v * n_heads)

        self.fc_o = nn.Linear(d_v * n_heads, hid_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # [batch, n head , length, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        G = torch.matmul(Q, K.permute(0, 1, 3, 2))

        G = G / (self.d_h**0.5)
        G += mask

        attention = torch.softmax(G, dim=-1)

        # attention = G
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_v * self.n_heads)
        x = self.fc_o(x)

        return x, G


class Block(nn.Module):
    def __init__(self, hid_dim, d_h, d_v, n_heads, pf_dim, pf_layers):
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(hid_dim, d_h, d_v, n_heads)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, hid_dim, pf_dim, pf_layers
        )

    def forward(self, src, mask=None):
        src, atten = self.self_attention(src, src, src, mask=mask)
        src = self.positionwise_feedforward(src)
        return src, atten


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        hid_dim,
        n_layers,
        d_h,
        d_v,
        n_heads,
        pf_dim,
        pf_layers,
        input_length,
    ):
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, hid_dim)
        self.input_mapping = PositionwiseFeedforwardLayer(hid_dim, hid_dim, pf_dim, pf_layers)

        self.output_mapping = nn.Linear(hid_dim, out_dim)
        self.layers = nn.ModuleList(
            [Block(hid_dim, d_h, d_v, n_heads, pf_dim, pf_layers) for _ in range(n_layers)]
        )

        self.position_embeddings = nn.Parameter(torch.rand(1, input_length, hid_dim))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

        # zero all the bias

        mask = self._generate_square_subsequent_mask(input_length)
        self.register_buffer("mask", mask)

        self.apply(init_weights)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src):
        src = self.input_embedding(src)
        # src = src + self.position_embeddings
        src = self.input_mapping(src)
        attens = []

        for layer in self.layers:
            src, atten = layer(src, mask=self.mask)
            attens.append(atten)

        output = self.output_mapping(src)

        return output, attens


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.register_buffer("scale", scale)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask="causal"):
        query, key, value = x, x, x

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = (
            query.permute(1, 0, 2)
            .contiguous()
            .view(len_q, batch_size * self.n_heads, self.head_dim)
        )
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask == "causal":
            mask = self._generate_square_subsequent_mask(len_q).to(query.device)
            # attn = attn.masked_fill(mask == 0, -1e10)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = (
            attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        )
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = (
            weight2.transpose(0, 1)
            .contiguous()
            .view(batch_size, self.n_heads, len_q, self.head_dim)
        )

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x
