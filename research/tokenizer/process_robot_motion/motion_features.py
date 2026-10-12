import torch
import torch.nn as nn
from torch import tensor
import numpy as np
import torch.nn.functional as F

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            
        

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 1,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self , seq_len ):
        super().__init__()
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(3, 3 * 3, bias=True )
        # output projection
        self.c_proj = nn.Linear(3, 3, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(0.2)
        self.resid_dropout = nn.Dropout(0.2)
        self.n_head =1 #config.n_head
        self.n_embd  = 3 #config.n_embd
        self.dropout = 0.2#config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len , seq_len ) )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self ):
        super().__init__()
        self.c_fc    = nn.Linear(3 , 4 * 3, bias=True )
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * 3 , 3 , bias= True )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self , seq_len ):
        super().__init__()
        self.ln_1 = LayerNorm( 3 , bias= True)
        self.attn = CausalSelfAttention(seq_len)
        self.ln_2 = LayerNorm(3 , bias= True )
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device , output_size=1,use_noise=None):
        super(MotionDiscriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise
        self.device = device

        # self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 3) 

        self.sig = nn.Sigmoid()

        n_layer = 6

        self.trans = nn.ModuleList([Block(180) for _ in range( n_layer)])
        self.trans = nn.Sequential( *self.trans )

        self.trans_text = nn.ModuleList([Block(12) for _ in range( n_layer)])
        self.trans_text = nn.Sequential( *self.trans_text )

        self.pos_emb_traj = nn.Embedding(180 , 3 ).to(device)
        self.pos_emb_text = nn.Embedding( 12 , 3 ).to(device)
        self.emb_text = nn.Embedding( 128 , 3 ).to(device)

        self.traj_pos_idx = torch.arange(0, 180, dtype=torch.long,  device=device)
        self.text_pos_idx = torch.arange(0, 12, dtype=torch.long,  device=device)

        # self.encoder = Encoder()
        # self.activation = nn.ReLU()

    def forward(self, motion_sequence, data_type, hidden_unit=None):

        if data_type == "motion":

            x = motion_sequence#.permute(0,2,1)
            b, t , _ = x.size()
            
            out = self.trans(x.float() +  self.pos_emb_traj(self.traj_pos_idx ) )

            out = self.linear1( out[:, -1 , : ] )
        
        if data_type == "text":
            idx = motion_sequence
            b, t = idx.shape

            tok_emb = self.emb_text(idx)
            pos_tok = self.pos_emb_text( self.text_pos_idx ) 

            out = self.trans_text( tok_emb + pos_tok )
            out = self.linear2( out[:, -1 , : ] )

        # x = self.encoder(x.float() ).squeeze(1)
        # x = self.linear1(x)  
        # out = self.sig (self.linear2(x))

        # dim (motion_length, num_samples, hidden_size)
        # if hidden_unit is None:
        #     motion_sequence = motion_sequence.permute(1, 0, 2)
        #     hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        # gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # # dim (num_samples, 30)
        # lin1 = self.linear1(gru_o[-1, :, :])
        # lin1 = torch.tanh(lin1)
        # # dim (num_samples, output_size)
        # lin2 = self.linear2(lin1).unsqueeze(1) 

        # lin3  = lin2.repeat( ( 1, 12 , 1 ) )

        # new_weights = self.concept_parameter.weight.repeat( ( lin3.shape[0] ,1 ,1 ) )

        # M1 = lin3 * new_weights
        # S1 = torch.linalg.norm( new_weights , dim =-1 ).unsqueeze(2)
        # S2 = torch.linalg.norm(  lin3 , dim =-1 ).unsqueeze(2)

        # lin4 = self.slope *torch.sum( (lin3 * new_weights) / ( S1*S2) , dim =-1) #torch.cat( [ lin3, new_weights] , dim =-1 )

        # # val = torch.sum ( lin4.permute(0,2,1) ,dim=1) #self.conv1d(lin4.permute(0,2,1)).squeeze(1)
        # out =  self.sig( lin4 )

        return out 
    
    def inference_forward( self, motion_sequence , data_type ):
        
        if data_type == "motion":
    
            x = motion_sequence#.permute(0,2,1)
            b, t , _ = x.size()
            traj_pos_idx2 = torch.arange(0, t , dtype=torch.long,  device="cuda")  
            out = self.trans(x.float() +  self.pos_emb_traj(traj_pos_idx2 ) )
            out = self.linear1( out[:, -1 , : ] )

        if data_type == "text":
            idx = motion_sequence
            b, t = idx.shape
            traj_pos_idx2 = torch.arange(0, t , dtype=torch.long,  device="cuda")  

            tok_emb = self.emb_text(idx)
            pos_tok = self.pos_emb_text( self.text_pos_idx ) 

            out = self.trans_text( tok_emb + pos_tok )
            out = self.linear2( out[:, -1 , : ] )
      
        return out 


    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)
