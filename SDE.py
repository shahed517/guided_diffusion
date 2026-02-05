import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F



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
        """Forward diffusion step for SDE-based diffusion model.

        Args:
            x0: (N, C, H, W) or (N, C, L)
            t: (N,)
        """
        time = t
        while time.ndim < x0.ndim:
            time = time.unsqueeze(-1)
        alpha_bar = self.get_signal_variance(time) # alpha_bar
        mean = x0*torch.sqrt(alpha_bar)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(1-alpha_bar)
        return xt, z
    
    def get_SNR(self, t: torch.tensor):
        """Signal-to-Noise Ratio (SNR) at time t, defined as alpha_bar/(1-alpha_bar)"""
        alpha_bar = self.get_signal_variance(t)
        return alpha_bar/(1-alpha_bar)

class SDEDiffusion(nn.Module, DiffusionBase):
    """Stochastic Differential Equation (SDE) based diffusion model, as proposed in 
    Song et al. (2021) 'SCORE-BASED GENERATIVE MODELING THROUGH SDEs' (All stars 2021)
    and used by 
    Popov et al. (2021) 'Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech' 
    t \in [0, T].
    """
    def __init__(self, estimator=None, beta_min=0.05, beta_max=20, T=1):
        super().__init__()
        DiffusionBase.__init__(self, estimator, beta_min, beta_max, T)


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

        alpha_bar = self.get_signal_variance(time) # alpha_bar
        noise_estimation = self.estimator(xt, t)
        noise_estimation *= torch.sqrt(1.0 - alpha_bar)
        loss = torch.sum((noise_estimation + z)**2) / (x0.numel())
        return loss, xt

    def compute_loss(self, x0, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, t)
    
    @torch.no_grad()
    def reverse_diffusion(self, z, n_timesteps=100, start_t=1.0, stoc=False):
        h = start_t / n_timesteps
        xt = z #* mask
        for i in range(n_timesteps):
            t = (start_t - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t
            while time.ndim < z.ndim:
                time = time.unsqueeze(-1)
            noise_t = self.get_noise(time)
            if stoc:  # adds stochastic term
                dxt_det = -0.5 * xt - self.estimator(xt,t)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                # print(xt.shape, t.shape)
                dxt = 0.5 * ( - xt - self.estimator(xt, t.view(z.shape[0])))
                dxt = dxt * noise_t * h
            xt = (xt - dxt)
        return xt

    @torch.no_grad()
    def reverse_diffusion_step(self, z, t, h, stoc=False):
        """One step of reverse diffusion at time t."""
        xt = z
        time = t
        while time.ndim < z.ndim:
            time = time.unsqueeze(-1)
        noise_t = self.get_noise(time)
        if stoc:  # adds stochastic term
            dxt_det = -0.5 * xt - self.estimator(xt,t)
            dxt_det = dxt_det * noise_t * h
            dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                    requires_grad=False)
            dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
            dxt = dxt_det + dxt_stoc
        else:
            dxt = 0.5 * ( - xt - self.estimator(xt, t))
            dxt = dxt * noise_t * h
        xt = (xt - dxt)
        return xt
    
    
                                            

    
    

        
