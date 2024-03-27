import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionLayer, self).__init__()
        self.dim = dim
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, vision_repr, text_repr):
        batch_size = vision_repr.size(0)
        query = self.query_projection(vision_repr).view(batch_size, -1, self.dim)
        key = self.key_projection(text_repr).view(batch_size, -1, self.dim)
        value = self.value_projection(text_repr).view(batch_size, -1, self.dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = attention_scores / (self.dim ** 0.5)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        cross_attended_repr = torch.bmm(attention_probs, value)
        cross_attended_repr = self.output_projection(cross_attended_repr.view(batch_size, -1, self.dim))

        return cross_attended_repr
