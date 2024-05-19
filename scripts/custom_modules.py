import torch.nn as nn

class ImagePathway(nn.Module):
    def __init__(self, original_layer, bottleneck, other_layers=None):
        super().__init__()
        layers = [original_layer]
        if other_layers:
            layers.extend(other_layers)
        layers.append(bottleneck)
        self.pathway = nn.Sequential(*layers)

    def forward(self, image_repr):
        return self.pathway(image_repr)

class TextPathway(nn.Module):
    def __init__(self, original_layer, bottleneck, other_layers=None):
        super().__init__()
        layers = [original_layer]
        if other_layers:
            layers.extend(other_layers)
        layers.append(bottleneck)
        self.pathway = nn.Sequential(*layers)

    def forward(self, text_repr):
        return self.pathway(text_repr)

class CrossAttentionModule(nn.Module):
    def __init__(self, image_pathway, text_pathway, cross_attention):
        super().__init__()
        self.image_pathway = image_pathway
        self.text_pathway = text_pathway
        self.cross_attention = cross_attention

    def forward(self, image_repr, text_repr):
        image_repr = self.image_pathway(image_repr)
        text_repr = self.text_pathway(text_repr)
        cross_attended_image, cross_attended_text = self.cross_attention(text_repr, image_repr)
        return cross_attended_image, cross_attended_text
