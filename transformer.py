'''
Implementation of transformer
Attention Is All You Need<https://arxiv.org/abs/1706.03762>
'''
import torch.nn as nn
import torch

class scaled_dot_attention(nn.Module):
    
    def __init__(self, dropout = 0.2):
        super(scaled_dot_attention, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, K, V, Q, mask = None):
        dim_len = Q.size(-1)
        assert dim_len == K.size(-1)
        
        weight_matrix = Q @ K.transpose(1, 2)
        scale = dim_len ** (1 / 2)
        scaled_weight_matrix = weight_matrix / scale
        if mask is not None:
            scaled_weight_matrix.masked_fill_(mask, float('-inf'))
        scaled_weight_matrix = self.dropout(scaled_weight_matrix)
        scaled_weight_matrix = self.softmax(scaled_weight_matrix)
        weighted_value_matrix = scaled_weight_matrix @ V
        return weighted_value_matrix

