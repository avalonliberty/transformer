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
        #Type : (Tensor, Tensor, Tensor, Tensor[Optional])
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

class attention_head(nn.Module):
    
    def __init__(self, d_model, d_feature):
        #Type : (int, int)
        super(attention_head, self).__init__()
        self.K_transform = nn.Linear(d_model, d_feature)
        self.V_transform = nn.Linear(d_model, d_feature)
        self.Q_transform = nn.Linear(d_model, d_feature)
        self.match = scaled_dot_attention()
        
    def forward(self, key, value, query, mask = None):
        #Type : (Tensor, Tensor, Tensor, Tensor[Optional])
        K = self.K_transform(key)
        V = self.V_transform(value)
        Q = self.Q_transform(query)
        
        return self.match(K, V, Q, mask)
    
class multiple_attention_head(nn.Module):
    
    def __init__(self, d_model, d_feature, n_head):
        super(multiple_attention_head, self).__init__()
        self.attention_list = nn.ModuleList([attention_head(d_model, d_feature) for _ in range(n_head)])
        self.transform = nn.Linear(n_head * d_feature, d_model)
        
    def forward(self, key, value, query, mask = None):
        features = [attn_head(key, value, query, mask) for attn_head in self.attention_list]
        feature_map = torch.cat(features, dim = -1)
        output = self.transform(feature_map)
        return output
        