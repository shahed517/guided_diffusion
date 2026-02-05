from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleUNet
from SDE import SDEDiffusion
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),                # Scales to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Scales to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize your SDE model
estimator = SimpleUNet()
sde_model = SDEDiffusion(estimator=estimator, beta_min=0.1, beta_max=20.0, T=1.0)
optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-4)

def train_mnist_sde(sde_model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(sde_model.estimator.parameters(), lr=lr)
    loss_history = []
    
    sde_model.estimator.train()
    print(f"Starting training on {sde_model.device}...")

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, _ in pbar:
            images = images.to(sde_model.device)
            
            # 1. Sample continuous time t uniformly [0, T]
            # It's vital to sample t uniformly so the model learns all noise levels
            t = torch.rand((images.shape[0],), device=sde_model.device) * sde_model.T
            
            # 2. Compute Loss using your SDEDiffusion logic
            # This handles the forward SDE (sample_xt) and the score matching objective
            loss, _ = sde_model.loss_t(images, t)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Optional: Gradient clipping is helpful in SDEs to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(sde_model.estimator.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(epoch_losses))
        
        loss_history.append(np.mean(epoch_losses))

    # --- Plotting the Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='b', label='SDE Loss')
    plt.title("Score-Based Generative Model Training Loss (MNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("Denoising Score Matching Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("/home/ahmed348/diffusion_practice/loss_curve.png")
    plt.show()

    return loss_history

# Execute
history = train_mnist_sde(sde_model, dataloader)

# Start with pure noise
z = torch.randn(16, 1, 28, 28).to(sde_model.device)
# Use your SDE.py reverse diffusion loop
samples = sde_model.reverse_diffusion(z, n_timesteps=500, stoc=True)
# Visualize (scale back from [-1, 1] to [0, 1])
samples = (samples + 1) / 2