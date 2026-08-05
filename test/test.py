import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from x_transformers import ContinuousTransformerWrapper, Decoder,Encoder

from x_transformers.x_transformers import AttentionLayers


input_dim = 1
seq_len = 5

model = ContinuousTransformerWrapper(
    dim_in=input_dim,
    dim_out=input_dim,
    max_seq_len=seq_len,
    attn_layers = Encoder(
        # causal=True,
        dim = 512,
        depth = 1,
        heads = 1,
        rel_pos_bias = True, 
        rel_pos_max_distance=seq_len,
        rel_pos_num_buckets=seq_len
    )
    
)
x1 = torch.tensor([1,0,0,0,0]).float().unsqueeze(0).unsqueeze(-1)
x2 = torch.tensor([1,1,0,0,0]).float().unsqueeze(0).unsqueeze(-1)
x3 = torch.tensor([1,1,1,0,0]).float().unsqueeze(0).unsqueeze(-1)
model(x1)