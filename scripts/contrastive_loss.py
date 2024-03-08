import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.shape[0]
        device = features.device

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(batch_size, device=device)

        # Remove the similarity with itself from the similarity matrix
        logits = similarity_matrix - mask * 9e15

        # Compute the loss
        labels = torch.arange(batch_size, device=device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss

class ContrastiveLearner(nn.Module):
    def __init__(self, base_model, temperature=0.07):
        super(ContrastiveLearner, self).__init__()
        self.base_model = base_model
        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(self, inputs):
        outputs = self.base_model(inputs)
        loss = self.contrastive_loss(outputs)
        return loss