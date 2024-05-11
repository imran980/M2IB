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

class mySequential(nn.Sequential):
    def forward(self, text_repr, image_repr):
        for module in self._modules.values():
            print("text_repr shape:", text_repr.shape)
            print("image_repr shape:", image_repr.shape)
            print("image_repr datatype------------------------:", image_repr.dtype)
            print("text_repr datatype------------------------:", text_repr.dtype)
            if isinstance(module, CrossAttentionLayer):
                text_repr, image_repr = module(text_repr, image_repr)
                print("CrossAttentionLayer text_repr shape:", text_repr.shape)
                print("CrossAttentionLayer image_repr shape:", image_repr.shape)
                print("CrossAttentionLayer image_repr datatype------------------------:", image_repr.dtype)
                print("CrossAttentionLayer text_repr datatype------------------------:", text_repr.dtype)
            else:
                text_repr = module(text_repr)
                image_repr = module(image_repr)
                print("else text_repr shape:", text_repr.shape)
                print("else image_repr shape:", image_repr.shape)
                print("else image_repr datatype------------------------:", image_repr.dtype)
                print("else text_repr datatype------------------------:", text_repr.dtype)
        return text_repr, image_repr


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    print("calling replace_layer--------------------------")
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
            # Extract the first argument from *args (inputs)
            inputs = args[0] if args else None
            other_args = args[1:]  # Collect the remaining positional arguments

            other_repr = kwargs.pop('other_repr', None)

            # Call mySequential.forward with inputs and other_repr as positional arguments
            # and pass the remaining arguments as keyword arguments
            return self.module(inputs, other_repr, **{**dict(zip(['arg{}'.format(i) for i in range(len(other_args))], other_args)), **kwargs})
        else:
            return self._original_forward(*args, **kwargs)

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

    setattr(model, '_original_forward', model.forward)
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

