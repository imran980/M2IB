"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scripts.utils import replace_layer, normalize, mySequential
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, mean_image: np.ndarray, std_image: np.ndarray, mean_text: np.ndarray, std_text: np.ndarray, device=None):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std_image = torch.tensor(std_image, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean_image = torch.tensor(mean_image, dtype=torch.float, device=self.device, requires_grad=False)
        self.std_text = torch.tensor(std_text, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean_text = torch.tensor(mean_text, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha_image = nn.Parameter(torch.full((1, *self.mean_image.shape), fill_value=self.initial_value, device=self.device))
        self.alpha_text = nn.Parameter(torch.full((1, *self.mean_text.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.mean_image.shape[-1], num_heads=8, batch_first=True)
        self.reset_alpha()

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha_image.fill_(self.initial_value)
            self.alpha_text.fill_(self.initial_value)
        return self.alpha_image, self.alpha_text

    def forward(self, image_features, text_features, **kwargs):
        lamb_image = self.sigmoid(self.alpha_image)
        lamb_image = lamb_image.expand(image_features.shape[0], image_features.shape[1], -1)
        masked_mu_image = image_features * lamb_image
        masked_var_image = (1 - lamb_image) ** 2

        lamb_text = self.sigmoid(self.alpha_text)
        lamb_text = lamb_text.expand(text_features.shape[0], text_features.shape[1], -1)
        masked_mu_text = text_features * lamb_text
        masked_var_text = (1 - lamb_text) ** 2

        # Apply cross-modal attention
        attended_image, attended_text = self.cross_attn(masked_mu_image, masked_mu_text, masked_mu_text)

        self.buffer_capacity = self._calc_capacity(attended_image, masked_var_image) + self._calc_capacity(attended_text, masked_var_text)
        t_image = self._sample_t(attended_image, masked_var_image)
        t_text = self._sample_t(attended_text, masked_var_text)
        return (t_image, t_text)


class IBAInterpreter:
    def __init__(self, model, estim_image: Estimator, estim_text: Estimator, beta, steps=10, lr=1, batch_size=10, progbar=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.original_layer_image = estim_image.get_layer()
        self.original_layer_text = estim_text.get_layer()
        self.shape_image = estim_image.shape()
        self.shape_text = estim_text.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)
        self.progbar = progbar
        self.lr = lr
        self.train_steps = steps
        self.bottleneck = InformationBottleneck(estim_image.mean(), estim_image.std(), estim_text.mean(), estim_text.std(), device=self.device)
        self.sequential_image = mySequential(self.original_layer_image, self.bottleneck)
        self.sequential_text = mySequential(self.original_layer_text, self.bottleneck)

    def vision_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t = self._run_vision_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1)[1:]  # Discard the first because it's the CLS token
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = torch.nn.functional.interpolate(saliency, size=224, mode='bilinear')
        saliency = saliency.squeeze().cpu().detach().numpy()
        return normalize(saliency)

    def text_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t = self._run_text_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        saliency = normalize(saliency)
        return normalize(saliency)

    def _run_vision_training(self, text_t, image_t):
        replace_layer(self.model.vision_model, self.original_layer_image, self.sequential_image)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t, text_features=text_t, image_features=image_t)
        replace_layer(self.model.vision_model, self.sequential_image, self.original_layer_image)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    def _run_text_training(self, text_t, image_t):
        replace_layer(self.model.text_model, self.original_layer_text, self.sequential_text)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t, text_features=text_t, image_features=image_t)
        replace_layer(self.model.text_model, self.sequential_text, self.original_layer_text)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    def _train_bottleneck(self, text_t, image_t, text_features, image_features):
        batch = (text_features.expand(self.batch_size, -1), image_features.expand(self.batch_size, -1, -1, -1))
        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
        self.bottleneck.reset_alpha()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck", disable=self.progbar):
            optimizer.zero_grad()
            out_text = self.model.get_text_features(batch[0])
            out_image = self.model.get_image_features(batch[1])
            t_image, t_text = self.bottleneck(image_features, text_features)
            loss_c, loss_f, loss_t = self.calc_loss(outputs=(out_text, out_image), labels=(batch[0], batch[1]), t_image=t_image, t_text=t_text)
            loss_t.backward()
            optimizer.step(closure=None)
        return loss_c, loss_f, loss_t

    def calc_loss(self, outputs, labels, t_image, t_text):
        """ Calculate the combined loss expression for optimization of lambda """
        compression_term = self.bottleneck.buffer_capacity.mean()
        fitting_term_image = self.fitting_estimator(outputs[0], t_image).mean()
        fitting_term_text = self.fitting_estimator(outputs[1], t_text).mean()
        fitting_term = (fitting_term_image + fitting_term_text) / 2
        total = self.beta * compression_term - fitting_term
        return compression_term, fitting_term, total

