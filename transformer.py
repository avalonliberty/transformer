'''
Implementation of transformer
Attention Is All You Need<https://arxiv.org/abs/1706.03762>
'''
from blocks import encoder_block, decoder_block
import torch.nn as nn
        
class encoder(nn.Module):
    
    def __init__(self, d_model = 512, d_feature = 64, n_head = 8, d_fc = 2048, n_blocks = 6):
        #Type : (int, int, int, int, int)
        super(encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([encoder_block(d_model, d_feature, n_head, d_fc) for _ in range(n_blocks)])
        
    def forward(self, x):
        #Type : (Tensor)
        for block in self.encoder_blocks:
            x = block(x)
        return x
    
class decoder(nn.Module):
    
    def __init__(self, d_model = 512, d_feature = 64, n_head = 8, d_fc = 2048, n_blocks = 6):
        #Type : (int, int, int, int, int)
        super(decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([decoder_block(d_model, d_feature, n_head, d_fc) for _ in range(n_blocks)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        for block in self.decoder_blocks:
            x = block(x, encoder_output, source_mask, target_mask)
        return x