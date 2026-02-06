from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleUNet, PowerfulUNet
from SDE import SDEDiffusion
import torch, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),                # Scales to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Scales to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Initialize your SDE model
estimator = PowerfulUNet()
sde_model = SDEDiffusion(estimator=estimator, beta_min=0.1, beta_max=20.0, T=1.0).to(device)

def train_mnist_with_sampling(sde_model, dataloader, epochs=100, lr=2e-4):
    optimizer = torch.optim.Adam(sde_model.estimator.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, epochs + 1):
        sde_model.estimator.train()
        epoch_loss = 0
        
        for images, _ in dataloader:
            images = images.to(sde_model.device)
            
            # Use your compute_loss logic
            loss, _ = sde_model.compute_loss(images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # --- Every 10 Epochs: Generate and Save Samples ---
        if epoch % 10 == 0 or epoch == 1:
            print(f">>> Generating progress samples for epoch {epoch}...")
            sde_model.save_weights(f"/home/shahed/guided_diffusion/weights/mnist_model_{epoch}.pth")
            sde_model.estimator.eval()
            with torch.no_grad():
                # 1. Start with 10 random noise vectors
                z = torch.randn(20, 1, 28, 28).to(sde_model.device)
                
                # 2. Use your reverse diffusion function
                # We use 500 steps for quality
                samples = sde_model.reverse_diffusion(z, n_timesteps=500, stoc=1)
                
                # 3. Post-process to [0, 1]
                samples = (samples + 1.0) / 2.0
                samples = samples.clamp(0.0, 1.0).cpu().numpy()
                
                # 4. Plot and Save
                fig, axes = plt.subplots(2, 10, figsize=(15, 4))
                for i, ax in enumerate(axes.flat):
                    # MNIST is grayscale, so we use cmap='gray'
                    ax.imshow(samples[i, 0], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"Sample {i+1}")
                
                plt.suptitle(f"Epoch {epoch} Samples (Loss: {avg_loss:.4f})")
                plt.savefig(f"/home/shahed/guided_diffusion/samples/samples_after_epoch_{epoch}.png")
                plt.close() # Close to free up memory

    # --- Plotting the Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='b', label='SDE Loss')
    plt.title("Score-Based Generative Model Training Loss (MNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("Denoising Score Matching Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("/home/shahed/guided_diffusion/loss_curve.png")
    plt.show()
    return loss_history

if __name__== "__main__":
    # Execute
    history = train_mnist_with_sampling(sde_model, dataloader)