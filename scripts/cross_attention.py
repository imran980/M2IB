import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super().__init__()
        print("Cross attention layer- -------------:")
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn_weight = nn.Linear(hidden_dim, 1)

    def forward(self, text_features, image_features):
        projected_text = self.text_proj(text_features)  # [batch_size, text_len, hidden_dim]
        print("Cross attention layer projected_text- -------------:, projected_text")
        projected_image = self.image_proj(image_features)  # [batch_size, image_len, hidden_dim]
        print("Cross attention layer projected_image- -------------:", projected_image)

        attention_scores = self.attn_weight(torch.tanh(projected_text.unsqueeze(2) + projected_image.unsqueeze(1)))  # [batch_size, text_len, image_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, text_len, image_len]

        text_attended = torch.bmm(attention_scores.softmax(dim=-1), image_features)  # [batch_size, text_len, image_dim]
        image_attended = torch.bmm(attention_scores.transpose(1, 2).softmax(dim=-1), text_features)  # [batch_size, image_len, text_dim]

        return text_attended, image_attended
