"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scripts.cross_attention import CrossAttentionLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import types

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


class ImagePathway(nn.Module):
    def __init__(self, original_layer, bottleneck, other_layers):
        super(ImagePathway, self).__init__()
        self.sequential = nn.Sequential(
            original_layer,
            *other_layers,
            bottleneck
        )

    def forward(self, image_repr):
        return self.sequential(image_repr)

class TextPathway(nn.Module):
    def __init__(self, original_layer, bottleneck, other_layers):
        super(TextPathway, self).__init__()
        self.sequential = nn.Sequential(
            original_layer,
            *other_layers,
            bottleneck
        )

    def forward(self, text_repr):
        return self.sequential(text_repr)

class CrossAttentionModule(nn.Module):
    def __init__(self, image_pathway, text_pathway, cross_attention):
        super(CrossAttentionModule, self).__init__()
        self.image_pathway = image_pathway
        self.text_pathway = text_pathway
        self.cross_attention = cross_attention

    def forward(self, image_repr, text_repr):
        image_repr = self.image_pathway(image_repr)
        text_repr = self.text_pathway(text_repr)
        cross_attended_image, cross_attended_text = self.cross_attention(text_repr, image_repr)
        # Merge or combine the cross-attended representations as needed
        return cross_attended_image, cross_attended_text

def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    print(3, replacement)
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

    

class CosSimilarity:
    """ Target function """
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)
    
class ImageFeatureExtractor(torch.nn.Module):
    """ Image feature wrapper """
    def __init__(self, model):
        super(ImageFeatureExtractor, self).__init__()
        self.model = model
                
    def __call__(self, x):
        return self.model.get_image_features(x)

class TextFeatureExtractor(torch.nn.Module):
    """ Text feature wrapper """
    def __init__(self, model):
        super(TextFeatureExtractor, self).__init__()   
        self.model = model
                
    def __call__(self, x):
        return self.model.get_text_features(x)
    
def image_transform(t, height=7, width=7):
    """ Transformation for CAM (image) """
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, 1 :  , :].reshape(t.size(0), height, width, t.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def text_transform(t):
    """ Transformation for CAM (text) """
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, :  , :].reshape(t.size(0), 1, -1, t.size(2))
    return result

