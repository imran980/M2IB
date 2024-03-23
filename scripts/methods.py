"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

from scripts.iba import IBAInterpreter, Estimator
import numpy as np
import clip
import copy
import torch 
from scripts.clip_wrapper import ContrastiveCLIPWrapper
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, x, is_text=False):
    with torch.no_grad():
        if is_text:
            states = model.get_text_features(x, output_hidden_states=True)
        else:
            states = model.get_image_features(x, output_hidden_states=True)
        feature = states['hidden_states'][layer_idx + 1]  # +1 because the first output is embedding
        return feature

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


def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var_image, var_text, lr=1, train_steps=10, progbar=False):
    features_image = extract_feature_map(model.vision_model, layer_idx, image_t, is_text=False)
    features_text = extract_feature_map(model.text_model, layer_idx, text_t, is_text=True)
    layer_image = extract_bert_layer(model.vision_model, layer_idx)
    layer_text = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator_image = get_compression_estimator(var_image, layer_image, features_image)
    compression_estimator_text = get_compression_estimator(var_text, layer_text, features_text)
    reader = IBAInterpreter(model, compression_estimator_image, compression_estimator_text, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.vision_heatmap(features_text, features_image)

def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var_image, var_text, lr=1, train_steps=10):
    features_image = extract_feature_map(model.vision_model, layer_idx, image_t, is_text=False)
    features_text = extract_feature_map(model.text_model, layer_idx, text_t, is_text=True)
    layer_image = extract_bert_layer(model.vision_model, layer_idx)
    layer_text = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator_image = get_compression_estimator(var_image, layer_image, features_image)
    compression_estimator_text = get_compression_estimator(var_text, layer_text, features_text)
    reader = IBAInterpreter(model, compression_estimator_image, compression_estimator_text, beta=beta, lr=lr, steps=train_steps)
    return reader.text_heatmap(features_text, features_image)
