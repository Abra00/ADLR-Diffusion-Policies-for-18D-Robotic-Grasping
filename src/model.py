import torch
from torch import nn
from torch.nn import functional as F
from src.positional_embeddings import PositionalEmbedding

class ResBlock(nn.Module):
    """
    The 'Brain Cell'.
    1. Project 'cond' to match 'x' size.
    2. Add them together.
    3. Pass through MLP (Norm -> Linear -> GELU).
    4. Add Residual connection (Input + Output).
    """
    def __init__(self, hidden_size: int, cond_size: int, dropout: float = 0.1):
        super().__init__()

        # Project condition to 2x hidden_size (Scale + Shift)
        self.cond_proj = nn.Linear(cond_size, hidden_size*2)
        #self.norm = nn.LayerNorm(hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):

        residual = x
        #x = x + self.cond_proj(cond)
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        
        #Formula AdaLN: Norm(x) * (1 + scale) + shift
        x = self.norm(x)*(1 + scale) + shift
        
        x = self.ff(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + residual

class VoxelEncoder(nn.Module):
    """
    Uses a loop to dynamically build a 3D CNN.
    """
    def __init__(self, input_channels=1, start_channels=32, levels=4, emb_size=256):
        super().__init__()
        
        layers = []
        c_in = input_channels
        c_out = start_channels

        for _ in range(levels):
            layers.append(nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(c_out),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            c_in = c_out
            c_out = c_out*2

        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(c_in, emb_size)

    def forward(self, x):

        x = self.conv_stack(x)
        x = self.global_pool(x)
        x = x.flatten(1)

        return self.fc(x)

class DiffusionMLP(nn.Module):
    """
    Connects the Encoder, the Time embedding, and the ResBlock Brain.
    """
    def __init__(self, 
                 input_dim=19, 
                 hidden_size=512, 
                 num_layers=4,
                 emb_size=256,
                 dropout=0.1,
                 input_emb_dim=None,
                 scale=2500):
        super().__init__()
        
        self.input_emb_dim = input_emb_dim
        self.time_mlp = PositionalEmbedding(emb_size, "sinusoidal")

        self.voxel_encoder = VoxelEncoder(
            input_channels=1,
            start_channels= 32,
            levels=4,
            emb_size=emb_size)

        if input_emb_dim is not None:
            # OPTION A: High Precision Mode
            # We use Sinusoidal embeddings, but control the size with 'input_emb_dim'
            self.input_mlp = PositionalEmbedding(input_emb_dim, "sinusoidal", scale=scale)
            print(f"using inp_emb_dim {input_emb_dim}")
            
            # The input to the main network grows: 19 * 32 = 608
            self.input_proj = nn.Linear(input_dim * input_emb_dim, hidden_size)
        else:
            # OPTION B: Simple Linear Mode
            self.input_proj = nn.Linear(input_dim, hidden_size)

        self.cond_size = emb_size + emb_size

        self.blocks = nn.ModuleList([
            ResBlock(hidden_size=hidden_size, cond_size=self.cond_size, dropout=dropout) 
            for _ in range(num_layers)])
        
        self.final = nn.Linear(hidden_size, input_dim)

    def forward(self, x, t, voxel_grid):

        t_emb = self.time_mlp(t)
        v_emb = self.voxel_encoder(voxel_grid)
        cond = torch.cat([t_emb, v_emb], dim=1)

        if self.input_emb_dim is not None:
            # Flatten the sinusoidal features: (B, 19, 32) -> (B, 608)
            x_emb = self.input_mlp(x).flatten(1)
            hidden_x = self.input_proj(x_emb)
        else:
            # Standard Linear
            hidden_x = self.input_proj(x)


        for block in self.blocks:
            hidden_x = block(hidden_x, cond)
            
        return self.final(hidden_x)