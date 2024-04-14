"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scripts.utils import replace_layer, normalize, mySequential
from scripts.cross_attention import CrossAttentionLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pdb

class Estimator:
    """
    Useful to calculate the empirical mean and variance of intermediate feature maps.
    """
    def __init__(self, layer):
        self.layer = layer
        self.M = None  # running mean for each entry
        self.S = None  # running std for each entry
        self.N = None  # running num_seen for each entry
        self.num_seen = 0  # total samples seen
        self.eps = 1e-5

    def feed(self, z: np.ndarray):

        # Initialize if this is the first datapoint
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1

        diff = (z - self.M)
        self.N += 1
        self.M += diff / self.num_seen
        self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M.squeeze()

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)  # Adding 1 for stablility, so that p_zero > 0 everywhere

    def std(self, stabilize=True):
        if stabilize:
            # Add small numbers, so that dead neurons are not a problem
            return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))

        else:
            return np.sqrt(self.S / self.N)

    def estimate_density(self, z):
        z_norm = (z - self.mean()) / self.std()
        p = z_norm.pdf(z_norm, 0, 1)
        return p

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        # Check if estimator classes match
        if self.__class__.__name__ != state["class"]:
            raise RuntimeError("This Estimator is {}, cannot load {}".format(self.__class__.__name__, state["class"]))
        # Check if layer classes match
        if self.layer.__class__.__name__ != state["layer_class"]:
            raise RuntimeError("This Layer is {}, cannot load {}".format(self.layer.__class__.__name__, state["layer_class"]))
        self.N = state["N"]
        self.S = state["S"]
        self.M = state["M"]
        self.num_seen = state["num_seen"]


class InformationBottleneck(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, device=None):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu, noise_var):
        #log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = noise_var.sqrt()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, var):
        # KL[P(t|x)||Q(t)] where Q(t) is N(0,1)
        kl =  -0.5 * (1 + torch.log(var) - mu**2 - var)
        return kl

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x, **kwargs):
        batch_size, seq_len, feature_dim = x.shape
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(batch_size, seq_len, feature_dim)
        print("IBAInterpreter lamb---------------------------", lamb)
        masked_mu = x * lamb
        masked_var = (1 - lamb) ** 2
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
        t = self._sample_t(masked_mu, masked_var)
        return (t,)


class IBAInterpreter:
    def __init__(self, model, estim: Estimator, beta, steps=10, lr=1, batch_size=10, progbar=False, dim_model=512):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)
        self.progbar = progbar
        self.lr = lr
        self.train_steps = steps
        self.bottleneck = InformationBottleneck(estim.mean(), estim.std(), device=self.device)
        self.cross_attention = CrossAttentionLayer(dim_model)
        self.sequential = mySequential(self.original_layer, self.bottleneck, self.cross_attention)



    def text_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t = self._run_text_training(text_t, image_t)     
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        saliency = normalize(saliency)
        return normalize(saliency)

    def vision_heatmap(self, text_t, image_t):
        print("vision_heatmap text_t------------------------:", text_t)
        print("vision_heatmap image_t------------------------:", image_t)
        print("vision_heatmap text_t------------------------:", text_t.shape)
        print("vision_heatmap image_t------------------------:", image_t.shape)
        saliency, loss_c, loss_f, loss_t = self._run_vision_training(text_t, image_t)
        print("vision_heatmap vision training--------------------", saliency, loss_c, loss_f, loss_t)
        saliency = torch.nansum(saliency, -1)[1:]  # Discard the first because it's the CLS token
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = torch.nn.functional.interpolate(saliency, size=224, mode='bilinear')
        saliency = saliency.squeeze().cpu().detach().numpy()
        return normalize(saliency)

    def _run_text_training(self, text_t, image_t):
        text_repr = self.model.get_text_features(text_t)
        image_repr = self.model.get_image_features(image_t)
        cross_attended_text, cross_attended_image = self.cross_attention(image_repr, text_repr)
        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(cross_attended_text, cross_attended_image)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    def _run_vision_training(self, text_t, image_t):
        print("run_vision_training-----------------------:")
        print("text_t-----------------------:",text_t)
        print("image_t-----------------------:",image_t)
        text_repr = self.model.get_text_features(text_t)
        image_repr = self.model.get_image_features(image_t)
        print("text_repr-----------------------:",text_repr)
        print("image_repr-----------------------:",image_repr)
        cross_attended_vision, cross_attended_image = self.cross_attention(image_repr, text_repr)
        print("cross_attended_vision-----------------------:",cross_attended_vision)
        print("cross_attended_image-----------------------:",cross_attended_image)
        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(cross_attended_image, cross_attended_vision)
        print("loss_c-----------------------:",loss_c)
        print("loss_f-----------------------:",loss_f)
        print("loss_t-----------------------:",loss_t)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    def _train_bottleneck(self, cross_attended_text, cross_attended_vision):
        print("_train_bottleneck-------------------------------------------")
        batch_text = cross_attended_text.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        batch_vision = cross_attended_vision.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        print("batch text-----------------------:", batch_text)
        print("batch_vision-----------------------:", batch_vision)
        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
        self.bottleneck.reset_alpha()
    
        self.model.eval()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck", disable=not self.progbar):
            optimizer.zero_grad()
            print("train bottleneck for loop-------------------------------------")
            # Call the forward method of the InformationBottleneck
            bottleneck_output_text = self.bottleneck(batch_text)[0]
        
            print("bottleneck_output_text-------------------:",bottleneck_output_text)
            bottleneck_output_vision = self.bottleneck(batch_vision)[0]
            print("bottleneck_output_vision-------------------:",bottleneck_output_vision)
            loss_c, loss_f, loss_t = self.calc_loss(outputs=bottleneck_output_text, labels=bottleneck_output_vision)
            loss_t.backward()
            optimizer.step(closure=None)
        print("loss_c-----------------------:",loss_c)
        print("loss_f-----------------------:",loss_f)
        print("loss_t-----------------------:",loss_t)
        return loss_c, loss_f, loss_t

        
        
    def calc_loss(self, outputs, labels):
        """ Calculate the combined loss expression for optimization of lambda """
        compression_term = self.bottleneck.buffer_capacity.mean()
        fitting_term = self.fitting_estimator(outputs, labels).mean()
        total =  self.beta * compression_term - fitting_term
        return compression_term, fitting_term, total
