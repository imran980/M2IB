import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.cross_attention = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)

    def forward(self, vision_repr, text_repr):
        cross_attended_repr, _ = self.cross_attention(vision_repr, text_repr, text_repr)
        return cross_attended_repr
