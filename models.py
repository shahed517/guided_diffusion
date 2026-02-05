import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Time embedding to help the network understand 't'
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        
        # Extremely simplified U-Net layers
        self.conv_in = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        # x: [Batch, 1, 28, 28], t: [Batch, 1]
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv_in(x)
        h = h + t_emb # Inject time information
        return self.conv_out(h)