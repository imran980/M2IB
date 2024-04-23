"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPVisionTransformer, CLIPTextTransformer
from scripts.cross_attention import CrossAttentionLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import types

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class mySequential(nn.Sequential):
    def forward(self, inputs, other_repr=None, **kwargs):
        if isinstance(inputs, tuple):
            print("MySequential: Receiving input shape:", inputs[0].shape)
            print("MySequential datatype------------------------:", inputs[0].dtype)
        else:
            print("MySequential: Receiving input shape:", inputs.shape)
            print("MySequential datatype------------------------:", inputs.dtype)

        for module in self._modules.values():
            if isinstance(module, CrossAttentionLayer):
                if other_repr is None:
                    raise ValueError("Cross-attention layer requires 'other_repr' input.")
                inputs, other_repr = module(inputs, other_repr)
            else:
                if isinstance(inputs, tuple):
                    inputs = module(*inputs)
                else:
                    inputs = module(inputs)

        if isinstance(inputs, tuple):
            print("MySequential: Returning input shape:", inputs[0].shape)
            print("MySequential: datatype:", inputs[0].dtype)
        else:
            print("MySequential: Returning input shape:", inputs.shape)
            print("MySequential: datatype:", inputs[0].dtype)

        return inputs


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential) or isinstance(model, mySequential):
                    model[int(name)] = replacement
                else:
                    setattr(model, name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    def forward_wrapper(self, *args, **kwargs):
        if hasattr(self, 'module') and isinstance(self.module, mySequential):
            # Extract the first argument from *args (pixel_values)
            pixel_values = args[0] if args else None
            # Extract other arguments from **kwargs
            output_attentions = kwargs.get('output_attentions', None)
            output_hidden_states = kwargs.get('output_hidden_states', None)
            return_dict = kwargs.get('return_dict', None)
            # Call the forward method of mySequential with the correct arguments
            return self.module(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        else:
            # Call the original forward method of the wrapped module
            return self._original_forward(*args, **kwargs)

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

    # Store the original forward method
    setattr(model, '_original_forward', model.forward)
    # Wrap the forward method of the parent module
    setattr(model, 'forward', types.MethodType(forward_wrapper, model))
    

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

