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
        batch_size, sequence_length, embedding_size = inputs.size()
        print("cross attention inputs shape---------------", inputs.shape)
        print("cross attention inputs---------------", inputs)
        query = self.query(inputs)
        print("cross attention query---------------", query)
        key = self.key(inputs)
        print("cross attention key---------------", key)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float))

        attention_weights = self.softmax(attention_scores)
        print("cross attention attention_scores---------------", attention_scores)
        value = self.value(inputs)
        print("cross attention inputs---------------", value)
        cross_attended_vision = torch.matmul(attention_weights, value)
        cross_attended_text = torch.matmul(attention_weights.transpose(-2, -1), inputs)
        
        return cross_attended_vision, cross_attended_text
