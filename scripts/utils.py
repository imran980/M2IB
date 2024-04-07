"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cross_attention_layer import CrossAttentionLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class mySequential(nn.Sequential):
    def forward(self, _input, attention_mask=None, causal_attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False, **kwargs):
        for module in self._modules.values():
            if isinstance(module, YourActualClassName):
                vision_repr, text_repr = _input
                _input = module(vision_repr, text_repr)
            else:
                if type(_input) == tuple:
                    _input = module(*_input, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
                else:
                    _input = module(_input, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        return _input

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

