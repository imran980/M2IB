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

    def forward(self, text_repr, image_repr):
        print("CA  text_t------:",  text_repr.shape)
        print("CA image_t------:",  image_repr.shape)
        batch_size, embedding_size = text_repr.size()
        _, embedding_size = image_repr.size()
        
        text_repr = text_repr.to(self.query.weight.device)
        image_repr = image_repr.to(self.query.weight.device)
        
        # Add a sequence dimension of size 1 to both tensors
        text_repr = text_repr.unsqueeze(1)
        image_repr = image_repr.unsqueeze(1)
        text_repr = text_repr.to(torch.float32)
        image_repr = image_repr.to(torch.float32)

        # Text cross-attention
        print("CA Dimensions of text_t------:",  text_repr.shape)
        print("CA Dimensions of image_t------:",  image_repr.shape)
        text_query = self.query(text_repr)
        image_key = self.key(image_repr)
        image_value = self.value(image_repr)
        
        #cross_attention_scores = torch.matmul(text_query, image_key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float))
        cross_attention_scores = torch.matmul(text_query, image_key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32))

        cross_attention_weights = self.softmax(cross_attention_scores)
        cross_attended_text = torch.matmul(cross_attention_weights, image_value)

        # Image cross-attention
        image_query = self.query(image_repr)
        text_key = self.key(text_repr)
        text_value = self.value(text_repr)
        #cross_attention_scores = torch.matmul(text_query, image_key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float))
        cross_attention_scores = torch.matmul(image_query, text_key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32))        cross_attention_weights = self.softmax(cross_attention_scores)
        cross_attended_image = torch.matmul(cross_attention_weights, text_value)

        # Remove the sequence dimension of size 1
        cross_attended_text = cross_attended_text.squeeze(1)
        cross_attended_image = cross_attended_image.squeeze(1)

        return cross_attended_text, cross_attended_image
