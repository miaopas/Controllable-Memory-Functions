from torch import nn
import torch
import math
from torchtune.modules import RotaryPositionalEmbeddings

MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

class RPETransformerModel(nn.Module):
    def __init__(self, style, causal=True, **kwargs):
        super().__init__()
        style = style.lower()
        
        # Validate the requested model style
        if style not in MODEL_REGISTRY:
            raise ValueError(f"Unknown style '{style}'. Available styles: {list(MODEL_REGISTRY.keys())}")
        
        # Instantiate the requested model with kwargs
        self.model = MODEL_REGISTRY[style](**kwargs)
        self.causal = causal

    def forward(self, *args, **kwargs):
        # Pass causal flag explicitly if the model supports it
        if 'causal' in self.model.forward.__code__.co_varnames:
            kwargs['causal'] = self.causal
        
        return self.model(*args, **kwargs)



@register_model('relative')
class StandardRPETransformerModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hid_dim, 
                 num_layers, 
                 nhead, 
                 dim_feedforward,
                 dropout=0,
                 max_distance=128, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, output_dim)
        self.layers = nn.ModuleList([RelativeEncoderLayer(hid_dim, 
                                                  nhead, 
                                                  dim_feedforward,
                                                  dropout,max_distance) 
                                     for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, src, src_mask=None, causal=True):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
                
        #pos = [batch size, src len]
        
        src = self.input_proj(src)

        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask, causal)
            
        #src = [batch size, src len, hid dim]
        src = self.output_layer(src)
        return src




class RelativeEncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 max_distance
                 ):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = RPEMultiHeadAttentionLayer(hid_dim, n_heads,max_distance)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, causal=True):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src = self.self_attention(src, src, src, src_mask,causal)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_distance):
        super().__init__()
        self.num_units = num_units
        self.max_distance = max_distance
        self.embeddings_table = nn.Parameter(torch.Tensor(max_distance * 2 + 1, num_units))
        nn.init.uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance)
        final_mat = distance_mat_clipped + self.max_distance
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    

class RPEMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, max_distance):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_distance = max_distance

        self.relative_position_k = RelativePosition(self.head_dim, self.max_distance)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_distance)
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # self.dropout = nn.Dropout(dropout)

        # Registering scale as a buffer to handle device placement automatically
        self.register_buffer("scale", torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))

    def forward(self, query, key, value, mask=None, causal=True):
        batch_size = query.shape[0]
        len_q = query.shape[1]
        len_k = key.shape[1]
        len_v = value.shape[1]

        # Linear projections
        Q = self.fc_q(query)  # [batch size, src len, hid_dim]
        K = self.fc_k(key)    # [batch size, src len, hid_dim]
        V = self.fc_v(value)  # [batch size, src len, hid_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, len_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, len_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, len_v, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Content-based attention scores
        content_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch size, n_heads, len_q, len_k]

        # Relative position attention scores
        rel_pos_k = self.relative_position_k(len_q, len_k)  # [len_q, len_k, head_dim]
        rel_scores = torch.einsum("bhqd,qkd->bhqk", Q, rel_pos_k)  # [batch size, n_heads, len_q, len_k]

        # Combine content and relative scores
        attn_scores = (content_scores + rel_scores) / self.scale

        # **Apply causal mask here**
        if causal:
            causal_mask_matrix = torch.tril(torch.ones(len_q, len_k, device=attn_scores.device))
            attn_scores = attn_scores.masked_fill(causal_mask_matrix == 0, -1e9)

        # Apply input mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout

        # attn_scores = self.dropout(attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        

        # Weighted sum of values
        content_output = torch.matmul(attn_weights, V)

        # Relative position values
        rel_pos_v = self.relative_position_v(len_q, len_v)  # [len_q, len_v, head_dim]
        rel_output = torch.einsum("bhqk,qvd->bhqd", attn_weights, rel_pos_v)


        # Combine content and relative outputs
        output = content_output + rel_output
        
        # Reshape back to [batch size, src len, hid_dim]
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, len_q, self.hid_dim)
        
        # Final linear projection
        output = self.fc_o(output)
        
        return output


@register_model('t5')
class T5StyleTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, num_layers, nhead, dim_feedforward, max_distance, dropout=0, **kwargs):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, output_dim)
        self.layers = nn.ModuleList([
            T5EncoderLayer(hid_dim, nhead, dim_feedforward, dropout,max_distance)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, causal=True):
        src = self.input_proj(src)
        for layer in self.layers:
            src = layer(src, src_mask, causal)
        src = self.output_layer(src)
        return src

class T5EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, max_distance):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = T5MultiHeadAttentionLayer(hid_dim, n_heads, max_distance)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, causal=True):
        # Self-attention with relative position bias
        _src = self.self_attention(src, src, src, src_mask, causal)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # Feedforward
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class T5MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, max_distance):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.relative_position_bias = T5RelativePositionBias(n_heads, max_distance=max_distance )

    def forward(self, query, key, value, mask=None, causal=True):
        batch_size, seq_len, _ = query.size()

        # Project and reshape
        Q = self.fc_q(query).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention score calculation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Adding relative position bias (adjusted for heads)
        position_bias = self.relative_position_bias(seq_len, seq_len).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        attn_scores += position_bias

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores.device)).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply input mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over last dimension (key dimension)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hid_dim)

        # Final linear projection
        output = self.fc_o(output)
        return output

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Parameter(torch.zeros((num_heads, 2 * max_distance - 1)))
        nn.init.xavier_uniform_(self.relative_bias)

    def forward(self, length_q, length_k):
        distance = torch.arange(length_k)[None, :] - torch.arange(length_q)[:, None]
        distance = torch.clamp(distance, -self.max_distance + 1, self.max_distance - 1)
        distance += self.max_distance - 1  # Shift to positive index range
        return self.relative_bias[:, distance]
    

class ConstantPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class ParameterPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = nn.Parameter(torch.randn(max_len, d_model))
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

@register_model('absolute')
class TransformerModel(nn.Module):

    def __init__(self,
            input_dim,
            output_dim,
            num_layers,
            hid_dim,
            nhead,
            src_length,
            dim_feedforward,
            dropout=0.1,**kwargs):
        
        super().__init__()
        # hid_dim = hid_dim * nhead
        # hid_dim = hid_dim if hid_dim % 2 == 0 else hid_dim + 1

        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hid_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hid_dim, output_dim)
        
        # Causal Mask
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(src_length, src_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        
        x = self.transformer(x, mask=self.causal_mask)  # (seq_len, batch, d_model)
        
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.output_layer(x)  # (batch, seq_len, output_dim)
    


class RotaryMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1, max_distance=4096, base=10000):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # Correctly initialize the rotary embedding module
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_distance, base=base)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, causal=True):
        batch_size, seq_len, _ = query.size()

        # Project to heads
        Q = self.fc_q(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.fc_k(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.fc_v(value).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply rotary embeddings
        Q = self.rope(Q)  # Shape: [b, s, n_h, h_d]
        K = self.rope(K)  # Shape: [b, s, n_h, h_d]

        # Compute attention scores
        attn_scores = torch.einsum("bsnh,btnh->bnst", Q, K) / self.scale

        # Apply causal mask if needed
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_scores.device))
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply input mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over last dimension (key dimension)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        output = torch.einsum("bnst,btnh->bsnh", attn_weights, V)

        # Reshape back to [batch_size, seq_len, hid_dim]
        output = output.contiguous().view(batch_size, seq_len, self.hid_dim)

        # Final linear projection
        return self.fc_o(output)



class RotaryEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, max_distance=4096):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = RotaryMultiHeadAttentionLayer(hid_dim, n_heads, dropout, max_distance)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, causal=True):
        # Pre-layer norm for stability
        src_norm = self.self_attn_layer_norm(src)

        # Self-attention with rotary embeddings
        _src = self.self_attention(src_norm, src_norm, src_norm, src_mask, causal)
        src = src + self.dropout(_src)

        # Positionwise feedforward
        _src = self.positionwise_feedforward(self.ff_layer_norm(src))
        src = src + self.dropout(_src)

        return src


@register_model('rotary')
class RotaryTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, num_layers, nhead, dim_feedforward, max_distance=4096, dropout=0.1, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, output_dim)
        self.layers = nn.ModuleList([
            RotaryEncoderLayer(hid_dim, nhead, dim_feedforward, dropout, max_distance)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, causal=True):
        src = self.input_proj(src)
        for layer in self.layers:
            src = layer(src, src_mask, causal)
        src = self.output_layer(src)
        return src
