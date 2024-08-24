import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scripts.utils import replace_layer, normalize, mySequential
from scripts.cross_attention import CrossAttentionLayer
from scripts.loss_focal import FocalLoss

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

class GradientEnhancedIBAInterpreter(nn.Module):
    def __init__(self, model, estim, beta, steps=10, lr=1, batch_size=10, progbar=False, dim_model=512):
        super().__init__()
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
        self.bottleneck = GradientGuidedInformationBottleneck(estim.mean(), estim.std(), device=self.device)
        self.sequential = mySequential(self.original_layer, self.bottleneck)
        self.cross_attention = CrossAttentionLayer(dim_model)
        
        self.focal = FocalLoss(class_num=2, alpha=0.5, gamma=2.5, size_average=True)
        self.focal = self.focal.to(self.device)
        self.softmax = nn.Softmax(dim=1)
        
        self.temperature = 0.07
        self.vsd_loss_weight = 2.9
        self.focal_loss_weight = 3.2
        self.grad_weight = 1.0
        
    def text_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t, grad_saliency = self._run_text_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        saliency = normalize(saliency)
        grad_saliency = grad_saliency.cpu().detach().numpy()
        return saliency * grad_saliency

    def vision_heatmap(self, text_t, image_t):
        saliency, loss_c, loss_f, loss_t, grad_saliency = self._run_vision_training(text_t, image_t)
        saliency = torch.nansum(saliency, -1)[1:]  # Discard the first because it's the CLS token
        dim = int(saliency.numel() ** 0.5)
        saliency = saliency.reshape(1, 1, dim, dim)
        saliency = F.interpolate(saliency, size=224, mode='bilinear')
        saliency = saliency.squeeze().cpu().detach().numpy()
        grad_saliency = grad_saliency.cpu().detach().numpy()
        return normalize(saliency), grad_saliency 

    def _run_text_training(self, text_t, image_t):
        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t, grad_saliency = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t, grad_saliency

    def _run_vision_training(self, text_t, image_t):
        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t, grad_saliency = self._train_bottleneck(text_t, image_t)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t, grad_saliency

    def _train_bottleneck(self, text_t: torch.Tensor, image_t: torch.Tensor):
        batch = text_t.expand(self.batch_size, -1), image_t.expand(self.batch_size, -1, -1, -1)
        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
        self.bottleneck.alpha.data.fill_(self.bottleneck.initial_value)
        
        self.model.eval()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck", disable=not self.progbar):
            optimizer.zero_grad()
            out = self.model.get_text_features(batch[0]), self.model.get_image_features(batch[1])
            attended_image, attended_text = self.cross_attention(out[1], out[0])
            loss_c, loss_f, loss_t, grad_saliency = self.calc_loss(outputs=attended_text, labels=attended_image)
            loss_t.backward()
            optimizer.step()
        return loss_c, loss_f, loss_t, grad_saliency

    def calc_loss(self, outputs, labels):
        loss_c = self.bottleneck.buffer_capacity.mean()
        
        outputs = F.normalize(outputs, dim=-1)
        labels = F.normalize(labels, dim=-1)
        
        loss_f = self.fitting_estimator(outputs, labels).mean()
        
        vsd_loss = F.kl_div(
            F.log_softmax(outputs / self.temperature, dim=-1),
            F.softmax(labels / self.temperature, dim=-1),
            reduction='batchmean'
        )
        
        batch_size = outputs.shape[0]
        binary_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        binary_labels[:] = 1  # Assuming positive class
        
        focal_inputs = outputs.mean(dim=-1)
        focal_inputs = torch.stack((1 - focal_inputs, focal_inputs), dim=-1)
        
        focal_loss = self.focal(focal_inputs, binary_labels)
        
        # Calculate gradient-based saliency
        grad_saliency = self.calculate_gradient_saliency(outputs, labels)
        
        loss_t = (
            self.beta * loss_c
            - loss_f
            + self.vsd_loss_weight * vsd_loss
            + self.focal_loss_weight * focal_loss
            + self.grad_weight * grad_saliency.mean()
        )
        
        return loss_c, loss_f, loss_t, grad_saliency

    def calculate_gradient_saliency(self, outputs, labels):
        outputs.retain_grad()
        similarity = F.cosine_similarity(outputs, labels)
        similarity.backward(torch.ones_like(similarity), retain_graph=True)
        grad_saliency = outputs.grad.abs().mean(dim=-1)
        outputs.grad.zero_()
        return grad_saliency

class GradientGuidedInformationBottleneck(nn.Module):
    def __init__(self, mean, std, device=None):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

    def forward(self, x):
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1)
        
        masked_mu = x * lamb
        masked_var = (1-lamb)**2
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
        t = self._sample_t(masked_mu, masked_var)
        return t

    @staticmethod
    def _sample_t(mu, noise_var):
        noise_std = noise_var.sqrt()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, var):
        kl = -0.5 * (1 + torch.log(var) - mu**2 - var)
        return kl
