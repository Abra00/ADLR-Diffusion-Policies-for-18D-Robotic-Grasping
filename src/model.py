import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
from torch import nn
from torch.nn import functional as F
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class VoxelEncoder(nn.Module):
    def __init__(self, input_shape=(32,32,32), emb_size=128):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # -> (batch_size, 64, 16, 16, 16)
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64), # <-- ADDED
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (batch_size, 128, 8, 8, 8)
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (batch_size, 256, 4, 4, 4)
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool3d(1)
        )
        self.flatten = nn.Flatten()
        
        # Fully connected layer to get to the 19D latent vector
        self.fc = nn.Linear(256, emb_size)


    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
  


class MLPWithVoxel(nn.Module):
    """MLP, das Voxel-Embeddings, Zeit-Embeddings und evtl. andere Inputs kombiniert"""
    def __init__(self, hidden_size=256, hidden_layers=3, emb_size=128,
                 time_emb="sinusoidal", input_emb="sinusoidal", voxel_input_shape=(32,32,32)):
        super().__init__()
        # Encoder für Voxelgrids
        self.voxel_encoder = VoxelEncoder(input_shape=voxel_input_shape, emb_size=emb_size)
        # Zeit-Embedding
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        # 19 D Vector embedding 
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        # Gesamtgröße der concatenated Embeddings
        concat_size = (19*emb_size) + emb_size + emb_size

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 19))  # Output: Rauschvorhersage pro 19D
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, voxel_grid):
        # Embeddings berechnen
        x_emb = self.input_mlp(x)   
        x_emb = x_emb.flatten(start_dim=1) 
        t_emb = self.time_mlp(t)
        voxel_emb = self.voxel_encoder(voxel_grid)
        # Concatenate
        x_cat = torch.cat([x_emb, t_emb, voxel_emb], dim=-1)
        return self.joint_mlp(x_cat)


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 device=None):

        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device("cpu")
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # Move all precomputed tensors to the device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)

        self.sqrt_inv_alphas_cumprod = self.sqrt_inv_alphas_cumprod.to(self.device)
        self.sqrt_inv_alphas_cumprod_minus_one = self.sqrt_inv_alphas_cumprod_minus_one.to(self.device)

        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)


    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        timesteps = timesteps.to(self.device)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps