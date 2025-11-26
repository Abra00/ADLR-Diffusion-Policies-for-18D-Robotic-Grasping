import torch
from torch import nn
from torch.nn import functional as F
from src.positional_embeddings import PositionalEmbedding

class ResBlock(nn.Module):
    """
    Improved Block:
    1. Accepts 'cond' (Voxel + Time) to inject context at every layer.
    2. Uses Residual connection.
    """
    def __init__(self, size: int, cond_size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        # LayerNorm is generally better than BatchNorm for MLPs
        self.norm = nn.LayerNorm(size)
        
        # Projection layer to match condition size to hidden size
        self.cond_proj = nn.Linear(cond_size, size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # 1. Add Condition (Shift)
        # We project condition to match 'x' size and add it
        h = x + self.cond_proj(cond)
        
        # 2. Standard MLP Block
        h = self.norm(h)
        h = self.act(self.ff(h))
        
        # 3. Residual Connection
        return x + h

class VoxelEncoder(nn.Module):
    def __init__(self, input_shape=(32,32,32), emb_size=256):
        super().__init__()
        
        # Using GroupNorm for stability with small batch sizes
        self.conv_stack = nn.Sequential(
            # 32 -> 16
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4 -> 2 (Instead of 1x1x1, we keep some spatial resolution)
            nn.Conv3d(128, 256, 4, 2, 1), # Output is 2x2x2
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        
        # 256 channels * 2 * 2 * 2 = 2048 features
        # We compress this to emb_size
        self.fc = nn.Linear(256 * 2 * 2 * 2, emb_size)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class MLPWithVoxel(nn.Module):
    def __init__(self, hidden_size=512, hidden_layers=4, emb_size=256,
                 time_emb="sinusoidal", input_emb="sinusoidal", 
                 voxel_input_shape=(32,32,32), input_dim=19):
        super().__init__()
        
        # 1. Encoders
        self.voxel_encoder = VoxelEncoder(input_shape=voxel_input_shape, emb_size=emb_size)
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        # 2. The Main Network (ResNet style)
        # The input to the first layer is just the Noisy Vector Embedding
        self.input_proj = nn.Linear(input_dim * emb_size, hidden_size)
        
        # The conditioning vector will be (Time_Emb + Voxel_Emb)
        cond_size = emb_size + emb_size 

        self.blocks = nn.ModuleList([
            ResBlock(hidden_size, cond_size) for _ in range(hidden_layers)
        ])
        
        # 3. Output Head
        self.final = nn.Linear(hidden_size, input_dim)

    def forward(self, x, t, voxel_grid):
        # A. Embed Inputs
        x_emb = self.input_mlp(x).flatten(start_dim=1) # (B, 19*128)
        t_emb = self.time_mlp(t)                       # (B, 128)
        v_emb = self.voxel_encoder(voxel_grid)         # (B, 128)
        
        # B. Prepare Condition
        # Combine Time and Voxel into one "Context Vector"
        cond = torch.cat([t_emb, v_emb], dim=-1)       # (B, 256)
        
        # C. Main Loop
        h = self.input_proj(x_emb)
        
        for block in self.blocks:
            # We pass 'cond' into every block
            h = block(h, cond)
            
        return self.final(h)

# Note: NoiseScheduler remains the same, it is mathematically correct.
class NoiseScheduler():
    # ... (Copy your existing NoiseScheduler here) ...
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", device=None):
        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device("cpu")
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

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

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
        return pred_prev_sample + variance
    
    def get_variance(self, t):
        if t == 0: return 0
        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def add_noise(self, x_start, x_noise, timesteps):
        timesteps = timesteps.to(self.device)
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        return s1 * x_start + s2 * x_noise
    
    def __len__(self):
        return self.num_timesteps