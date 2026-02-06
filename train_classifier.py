import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from SDE import SDEDiffusion
from models import MNISTClassifier, PowerfulUNet   # only to load noise schedule if needed

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Dataset
# ------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

# ------------------------
# Load diffusion model (FORWARD PROCESS ONLY)
# ------------------------
# We only need this for sample_xt(), not training
dummy_estimator = PowerfulUNet().to(device)

sde = SDEDiffusion(
    estimator=dummy_estimator,
    beta_min=0.1,
    beta_max=20.0,
    T=1.0
).to(device)

sde.estimator.eval()
for p in sde.estimator.parameters():
    p.requires_grad_(False)

# ------------------------
# Classifier
# ------------------------
classifier = MNISTClassifier(base_ch=64).to(device)

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

# ------------------------
# Training hyperparameters
# ------------------------
epochs = 20
log_interval = 100

# ------------------------
# Training loop
# ------------------------
classifier.train()

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for i, (x0, y) in enumerate(pbar):
        x0 = x0.to(device)
        y = y.to(device)

        # Sample random diffusion time
        t = torch.rand(x0.shape[0], device=device)

        # Generate noisy images using YOUR forward SDE
        xt, _ = sde.sample_xt(x0, t)

        # Classifier prediction
        logits = classifier(xt, t)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if i % log_interval == 0:
            pbar.set_postfix({
                "loss": running_loss / (i + 1),
                "acc": 100.0 * correct / total
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_acc:.2f}%"
    )

# ------------------------
# Save classifier
# ------------------------
torch.save(
    classifier.state_dict(),
    "/home/shahed/guided_diffusion/weights/mnist_classifier.pth"
)

print("Classifier training complete. Weights saved.")
