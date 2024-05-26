"""
The code below wraps the openai clip model to faciliate extracting layers and encoders.
Based on https://github.com/openai/CLIP and 
"""
import copy
import torch
import torch.nn as nn
from functools import partial
from scripts.utils import ImagePathway, TextPathway, CrossAttentionModule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def permute_then_forward(self, x):
    x = x.permute(1, 0, 2)
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    x = x.permute(1, 0, 2) 
    return x

class VisionEmbeddings(nn.Module):
    def __init__(self, class_embedding, patch_embedding, positional_embedding, dtype):
        super().__init__()
        self.class_embedding = class_embedding
        self.patch_embedding = patch_embedding
        self.positional_embedding = positional_embedding
        self.dtype = dtype

    def forward(self, x):
        x = self.patch_embedding(x.to(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(self.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(self.dtype)
        return x

class image_encoder_wrapper(nn.Module):
    def __init__(self, model, dtype, cross_attention):
        super().__init__()
        self.transformer = model.transformer
        self.embeddings = model.embeddings
        self.ln_pre = model.ln_pre
        self.ln_post = model.ln_post
        self.proj = model.proj
        self.dtype = dtype
        self.cross_attention = cross_attention
        for layer in self.transformer.resblocks:
            layer.forward = partial(permute_then_forward, layer)

    def forward(self, x, emb_input=False, output_hidden_states=False):
        if not emb_input:
            x = self.embeddings(x)
        x = self.ln_pre(x).to(self.dtype)
        hidden_states = [x.clone().detach()]
        for layer in self.transformer.resblocks:
            x = layer(x.to(self.dtype))
            if type(x) == tuple and len(x) == 1: x = x[0]
            hidden_states.append(x.clone().detach())
        x = self.ln_post(x[:, 0, :]).type(self.dtype)
        if self.proj is not None:
            x = x @ self.proj
        cross_attended_image, _ = self.cross_attention(x, text_repr)
        if output_hidden_states:
            return {'pooler_output': cross_attended_image, 'hidden_states': hidden_states}
        else:
            return cross_attended_image

class TextEmbeddings(nn.Module):
    def __init__(self, token_embedding, positional_embedding, dtype):
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.dtype = dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)[:x.shape[1], :]#(1,50,512)
        return x

class text_encoder_wrapper(nn.Module):
    def __init__(self, model, cross_attention):
        super().__init__()
        self.transformer = model.transformer
        self.embeddings = TextEmbeddings(model.token_embedding, model.positional_embedding, model.dtype)
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = model.dtype
        self.cross_attention = cross_attention
        for layer in self.transformer.resblocks:
            layer.attn_mask = None
            layer.forward = partial(permute_then_forward, layer)

    def forward(self, x, image_repr, output_hidden_states=False, emb_input=False):
        maxidx = -1 #x.argmax(dim=-1) take features from the eot embedding (eot_token is the highest number in each sequence)
        if not emb_input:
            x = self.embeddings(x)
        hidden_states = [x.clone().detach()] # embedding output
        for layer in self.transformer.resblocks:
            x = layer(x.to(self.dtype))
            if type(x) == tuple and len(x) == 1: x = x[0]
            hidden_states.append(x.clone().detach())
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        x = x[torch.arange(x.shape[0]), maxidx] @ self.text_projection
        cross_attended_text, _ = self.cross_attention(x, image_repr)
        if output_hidden_states:
            return {'pooler_output': cross_attended_text, 'hidden_states': hidden_states}
        else:
            return cross_attended_text
            
class CLIPEncoderWrapper(nn.Module):
    def __init__(self, clip_model, layer_idx, dim_model):
        super().__init__()
        self.text_encoder = text_encoder_wrapper(clip_model)
        self.image_encoder = image_encoder_wrapper(clip_model, clip_model.dtype)
        self.cross_attention = CrossAttentionLayer(dim_model)
        self.layer_idx = layer_idx

    def forward(self, text_input, image_input):
        text_repr = self.text_encoder(text_input, emb_input=False, output_hidden_states=True)['hidden_states'][self.layer_idx + 1]
        image_repr = self.image_encoder(image_input, emb_input=False, output_hidden_states=True)['hidden_states'][self.layer_idx + 1]
        cross_attended_text, cross_attended_image = self.cross_attention(text_repr, image_repr)
        return cross_attended_text, cross_attended_image


from scripts.cross_attention import CrossAttentionLayer

class ClipWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual_model = model.visual
        self.text_model = model.transformer
        self.dim_model = model.visual.proj.weight.shape[1]
        self.cross_attention = CrossAttentionLayer(self.dim_model)

        # Wrap the models with custom encoders
        self.vision_model = image_encoder_wrapper(self.visual_model, model.visual.dtype, self.cross_attention)
        self.text_model = text_encoder_wrapper(self.text_model, self.cross_attention)

    def get_image_features(self, x, output_hidden_states=False, emb_input=False):
        text_repr = self.text_model(x, emb_input=emb_input, output_hidden_states=False)
        image_repr = self.vision_model(x, emb_input=emb_input, output_hidden_states=False)
        cross_attended_image, _ = self.cross_attention(image_repr, text_repr)
        if output_hidden_states:
            return {'pooler_output': cross_attended_image, 'hidden_states': []}
        else:
            return cross_attended_image

    def get_text_features(self, x, output_hidden_states=False, emb_input=False):
        text_repr = self.text_model(x, emb_input=emb_input, output_hidden_states=False)
        image_repr = self.vision_model(x, emb_input=emb_input, output_hidden_states=False)
        _, cross_attended_text = self.cross_attention(image_repr, text_repr)
        if output_hidden_states:
            return {'pooler_output': cross_attended_text, 'hidden_states': []}
        else:
            return cross_attended_text
