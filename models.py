import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Time embedding: Projects a single scalar 't' into a higher dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 64) # Match the number of channels in the hidden layer
        )
        
        self.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        # 1. Process time: [Batch] -> [Batch, 64, 1, 1]
        t_emb = self.time_mlp(t.view(-1, 1))
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        
        # 2. Process image
        h = F.relu(self.conv_in(x))
        
        # 3. Combine: The (1, 1) spatial dimensions of t_emb 
        # will automatically broadcast to the (28, 28) of h
        h = h + t_emb
        return self.conv_out(h)  

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        # If channel dims change, use a 1x1 conv for the skip
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = F.gelu(h)

        # Inject time embedding
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        h = self.conv2(h)
        return F.gelu(h + self.skip(x))


class PowerfulUNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.base_ch = base_ch
        time_dim = 64

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # -------- Encoder --------
        self.enc1 = ResBlock(1, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        # -------- Bottleneck --------
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        # -------- Decoder --------
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec3 = ResBlock(base_ch * 4, base_ch * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.view(-1, 1))

        # Encoder
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.enc3(self.down2(h2), t_emb)

        # Bottleneck
        h = self.mid1(h3, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder
        h = self.up3(h)
        h = self.dec3(torch.cat([h, h2], dim=1), t_emb)

        h = self.up2(h)
        h = self.dec2(torch.cat([h, h1], dim=1), t_emb)

        return self.out(h)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = F.gelu(self.conv1(x))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return F.gelu(h + self.skip(x))


class MNISTClassifier(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        time_dim = base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.block1 = ClassifierResBlock(1, base_ch, time_dim)
        self.block2 = ClassifierResBlock(base_ch, base_ch * 2, time_dim)
        self.block3 = ClassifierResBlock(base_ch * 2, base_ch * 4, time_dim)

        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 4, 10)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.view(-1, 1))

        h = self.block1(x, t_emb)
        h = self.down1(h)

        h = self.block2(h, t_emb)
        h = self.down2(h)

        h = self.block3(h, t_emb)

        h = self.pool(h).squeeze(-1).squeeze(-1)
        return self.fc(h)
