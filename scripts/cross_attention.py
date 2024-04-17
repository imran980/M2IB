import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_model):
        super(CrossAttentionLayer, self).__init__()
        self.dim_model = dim_model
        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        vision_repr, text_repr = inputs
        # ... (the rest of your code for the CrossAttentionLayer)
        # Calculate attention scores
        print("vision repr crossattentionlayer------------------:", vision_repr)
        print("text repr crossattentionlayer------------------:", text_repr)
        query = self.query(vision_repr)
        print("query crossattentionlayer------------------:", query)
        key = self.key(text_repr)
        print("key crossattentionlayer------------------:", key)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model))
        print("attention_scores crossattentionlayer------------------:", attention_scores)
        # Compute attention weights
        attention_weights = self.softmax(attention_scores)
        print("attention_weights crossattentionlayer------------------:", attention_weights)
        # Calculate cross-attended representations
        value = self.value(text_repr)
        cross_attended_vision = torch.matmul(attention_weights, value)
        cross_attended_text = torch.matmul(attention_weights.transpose(-2, -1), vision_repr)
        print("cross_attended_vision crossattentionlayer------------------:", cross_attended_vision)
        print("cross_attended_text crossattentionlayer------------------:", cross_attended_text)
        return cross_attended_vision, cross_attended_text
