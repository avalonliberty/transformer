'''
Implementation of basic block for transformation
'''
import torch.nn as nn
from utils import multiple_attention_head

class encoder_block(nn.Module):
    
    def __init__(self, d_model = 512, d_feature = 64, n_head = 8, d_fc = 2048):
        #Type : (int, int, int, int)
        super(encoder_block, self).__init__()
        self.attn_block = multiple_attention_head(d_model, d_feature, n_head)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p = 0.1)
        self.fc_block = nn.Sequential(
                nn.Linear(d_model, d_fc),
                nn.Dropout(p = 0.1),
                nn.Linear(d_fc, d_model)
                )
        
    def forward(self, x, mask = None):
        #Type : (Tensor)
        features = self.dropout(self.attn_block(x, x, x, mask))
        x = self.layerNorm1(x + features)
        feature = self.dropout(self.fc_block(x))
        features = self.layerNorm2(x + feature)
        return features
    
class decoder_block(nn.Module):
    
    def __init__(self, d_model = 512, d_feature = 64, n_head = 8, d_fc = 2048, source_mask, target_mask):
        super(decoder_block, self).__init__()
        self.masked_attn = multiple_attention_head(d_model, d_feature, n_head)
        self.decoder_encoder_attn = multiple_attention_head(d_model, d_feature, n_head)
        self.dropout = nn.Dropout(p = 0.1)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.layerNorm3 = nn.LayerNorm(d_model)
        self.fc_block = nn.Sequential(
                nn.Linear(d_model, d_fc),
                nn.Dropout(p = 0.1),
                nn.Linear(d_fc, d_model)
                )
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        #Type : (Tensor)
        feature = self.dropout(masked_attn(x, x, x, target_mask))
        x = self.layerNorm1(x + feature)
        feature = self.dropout(self.decoder_encoder_attn(encoder_output, encoder_output, x, source_mask))
        x = self.layerNorm2(x + feature)
        feature = self.dropout(self.fc_block(x))
        x = self.layerNorm3(x + feature)
        return x
        