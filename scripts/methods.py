"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

from scripts.iba import IBAInterpreter, Estimator
import numpy as np
import clip
import copy
import torch 
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, x, is_text=False):
    if is_text:
        text_features = model.get_text_features(x.long(), output_hidden_states=True)
        feature = text_features[1][layer_idx + 1]  # Access the hidden_states list directly
        return feature, text_features[0], None
    else:
        image_features, hidden_states = model.get_image_features(x, output_hidden_states=True)
        feature = hidden_states[layer_idx + 1]  # Access the hidden_states list directly
        text_features = model.get_text_features(x.long(), output_hidden_states=False)
        return feature, text_features, image_features

# Extract BERT Layer
def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'layers' or n == 'resblocks':
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        desired_layer = s2
                        return desired_layer

# Get an estimator for the compression term
def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var*np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator


def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10, progbar=True):
    features = extract_feature_map(model, layer_idx, text_t)
    layer = extract_bert_layer(model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.text_heatmap(text_t, image_t)

def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10, progbar=True):
    features = extract_feature_map(model, layer_idx, image_t, is_text=False)
    layer = extract_bert_layer(model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.vision_heatmap(text_t, image_t)
