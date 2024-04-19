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
       
        print("cross attention inputs shape---------------", inputs.shape)
        print("cross attention inputs---------------", inputs)
        batch_size, sequence_length, embedding_size = inputs.size()
        vision_repr = inputs[:, :, :embedding_size // 2]
        print("cross attention vision_repr shape---------------", vision_repr.shape)
        text_repr = inputs[:, :, embedding_size // 2:]
        print("cross attention text_repr shape---------------", text_repr.shape)
        query = self.query(vision_repr)
        print("cross attention query---------------", query)
        key = self.key(text_repr)
        print("cross attention key---------------", key)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float))
        print("cross attention attention_scores---------------", attention_scores)
        attention_weights = self.softmax(attention_scores)
        print("cross attention attention_weights---------------", attention_weights)
        value = self.value(text_repr)
        print("cross attention inputs---------------", value)
        cross_attended_vision = torch.matmul(attention_weights, value)
        cross_attended_text = torch.matmul(attention_weights.transpose(-2, -1), vision_repr)
        
        return cross_attended_vision, cross_attended_text
