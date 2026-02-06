import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Added for path handling

class DiffusionBase(ABC):
    def __init__(self, estimator, beta_min, beta_max, T):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = estimator
        if self.estimator is not None:
            self.estimator = self.estimator.to(self.device)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    @abstractmethod
    def get_signal_variance(self, t: torch.tensor):
        """For Ho.el al this is alpha_bar, for SDE this is torch.exp(-cum_noise)"""
        raise NotImplementedError
    
    def sample_xt(self, x0, t):
        """Forward diffusion step for SDE-based diffusion model."""
        time = t
        while time.ndim < x0.ndim:
            time = time.unsqueeze(-1)
        alpha_bar = self.get_signal_variance(time) 
        mean = x0*torch.sqrt(alpha_bar)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(1-alpha_bar)
        return xt, z
    
    def get_SNR(self, t: torch.tensor):
        """Signal-to-Noise Ratio (SNR) at time t"""
        alpha_bar = self.get_signal_variance(t)
        return alpha_bar/(1-alpha_bar)

class SDEDiffusion(nn.Module, DiffusionBase):
    def __init__(self, estimator=None, beta_min=0.1, beta_max=20, T=1):
        super().__init__()
        DiffusionBase.__init__(self, estimator, beta_min, beta_max, T)

    # --- NEW METHODS FOR SAVING AND LOADING ---

    def save_weights(self, path="sde_estimator.pth"):
        """Saves the estimator's state_dict to a file."""
        if self.estimator is not None:
            torch.save(self.estimator.state_dict(), path)
            print(f"Model weights saved to {path}")
        else:
            print("No estimator found to save.")

    def load_weights(self, path="sde_estimator.pth"):
        """Loads weights into the estimator from a file."""
        if self.estimator is not None:
            if os.path.exists(path):
                self.estimator.load_state_dict(torch.load(path, map_location=self.device))
                self.estimator.eval() # Set to evaluation mode for generation
                print(f"Model weights loaded from {path}")
            else:
                print(f"Error: Path {path} does not exist.")
        else:
            print("No estimator initialized to load weights into.")

    # --- REMAINING ORIGINAL METHODS ---

    def get_signal_variance(self, t):
        noise = self.beta_min*t + 0.5*(self.beta_max - self.beta_min)*(t**2)
        return torch.exp(-noise)
    
    def get_noise(self, t):
        noise = self.beta_min + (self.beta_max - self.beta_min)*t
        return noise
    
    def loss_t(self, x0, t):
        xt, z = self.sample_xt(x0, t)
        time = t
        while time.ndim < x0.ndim:
            time = time.unsqueeze(-1)

        alpha_bar = self.get_signal_variance(time) 
        # noise_estimation = self.estimator(xt, t) # the score
        # noise_estimation *= torch.sqrt(1.0 - alpha_bar) ## get the noise epsilon from score
        # loss = torch.sum((noise_estimation + z)**2) / (x0.numel())
        prediction = self.estimator(xt, t)
        loss = torch.mean((prediction - z)**2) # Simple MSE on noise
        return loss, xt

    def compute_loss(self, x0, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, t)
    
    @torch.no_grad()
    def reverse_diffusion(self, z, n_timesteps=100, start_t=1.0, stoc=1):
        h = start_t / n_timesteps
        xt = z 
        for i in range(n_timesteps):
            t = (start_t - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t
            while time.ndim < z.ndim: 
                time = time.unsqueeze(-1)
            noise_t = self.get_noise(time)
            alpha_bar = self.get_signal_variance(time)
            epsilon = self.estimator(xt, t)
            score = - epsilon / torch.sqrt(1 - alpha_bar)
            if stoc:  
                dxt_det = -0.5 * xt - score
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = (-0.5 * xt - score)
                dxt = dxt * noise_t * h
            xt = (xt - dxt)
        return xt
    
    def guided_reverse_diffusion(self, z, classifier, target_class, guidance_scale=3.0, n_timesteps=500, start_t=1.0, stoc=1):
        """
        Classifier-guided reverse diffusion.
        """
        h = start_t / n_timesteps
        xt = z
        for i in range(n_timesteps):
            t = (start_t - (i + 0.5) * h) * torch.ones(
                z.shape[0], device=z.device, dtype=z.dtype
            )
            time = t
            while time.ndim < xt.ndim:
                time = time.unsqueeze(-1)

            noise_t = self.get_noise(time)
            alpha_bar = self.get_signal_variance(time)
            # --------------------------------------------------
            # 1. Diffusion score (unconditional)
            # --------------------------------------------------
            xt.requires_grad_(True)
            epsilon = self.estimator(xt, t)
            score = -epsilon / torch.sqrt(1.0 - alpha_bar)
            # --------------------------------------------------
            # 2. Classifier guidance term
            # --------------------------------------------------
            logits = classifier(xt, t)
            log_probs = torch.log_softmax(logits, dim=1)
            selected = log_probs[:, target_class].sum()
            grad = torch.autograd.grad(selected, xt)[0]
            # Combine scores
            guided_score = score + guidance_scale * grad
            # --------------------------------------------------
            # 3. Reverse SDE step (Euler–Maruyama)
            # --------------------------------------------------
            if stoc:
                dxt_det = (-0.5 * xt - guided_score) * noise_t * h
                dxt_stoc = torch.randn_like(xt) * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = (-0.5 * xt - guided_score) * noise_t * h
            xt = (xt - dxt).detach()
        return xt
