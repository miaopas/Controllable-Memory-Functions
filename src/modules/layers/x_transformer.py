import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from x_transformers import ContinuousTransformerWrapper, Decoder,Encoder
from x_transformers.x_transformers import AttentionLayers


class ContinuousTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, depth, n_head, seq_len):
        super(ContinuousTransformer, self).__init__()
  
        self.model = ContinuousTransformerWrapper(
                    dim_in=input_dim,
                    dim_out=output_dim,
                    max_seq_len=seq_len,
                    attn_layers = Decoder(
                        dim =  hid_dim,
                        depth = depth,
                        heads = n_head,
                        rel_pos_bias = True, 
                        rel_pos_max_distance=seq_len,
                        rel_pos_num_buckets=seq_len,
                        ff_mult=1
                    )
                    
                )
       

    def forward(self, x):
        return self.model(x)
