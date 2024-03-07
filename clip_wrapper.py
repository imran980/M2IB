import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the contrastive loss function (e.g., InfoNCE loss)
def contrastive_loss(features, temperature=0.07):
    batch_size, _ = features.shape
    device = features.device

    # Compute the similarity matrix
    sim_matrix = torch.einsum('ik,jk->ij', features, features.detach()) / temperature
    sim_matrix.fill_diagonal_(-5e4)

    # Compute the loss
    negatives = sim_matrix.exp().sum(dim=-1)
    positives = torch.exp(torch.diagonal(sim_matrix))
    loss = -torch.log(positives / negatives).mean()

    return loss

# Update the forward method of your ClipWrapper
def forward(self, images, texts, augmented_images=None, augmented_texts=None):
    # Get the original image and text features
    image_features = self.get_image_features(images)
    text_features = self.get_text_features(texts)

    # Compute the cosine similarity loss between image and text features
    cosine_loss = 1 - F.cosine_similarity(image_features, text_features).mean()

    if augmented_images is not None and augmented_texts is not None:
        # Get the augmented image and text features
        augmented_image_features = self.get_image_features(augmented_images)
        augmented_text_features = self.get_text_features(augmented_texts)

        # Concatenate the original and augmented features
        all_features = torch.cat([image_features, text_features, augmented_image_features, augmented_text_features], dim=0)

        # Compute the contrastive loss
        contrastive_loss = contrastive_loss(all_features)

        # Combine the cosine similarity loss and contrastive loss
        total_loss = cosine_loss + contrastive_loss

        return total_loss
    else:
        return cosine_loss