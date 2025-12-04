import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    """
    Encodes a 3D SDF grid (32x32x32) into a 19-dimensional latent vector.
    
    This architecture is a 3-layer 3D convolutional stack.
    Each layer halves the spatial dimensions (e.g., 32 -> 16 -> 8 -> 4)
    and doubles the feature channels.
    """
    def __init__(self, latent_dim=19):
        super(Encoder3D, self).__init__()
        
        # Input: (batch_size, 1, 32, 32, 32)
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

            # Change the number of params from 2 million to 32 thousands
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Flatten the (256, 4, 4, 4) tensor
        # 256 * 4 * 4 * 4 = 16384
        #self.flatten = nn.Flatten()
        
        # Fully connected layer to get to the 19D latent vector
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, 1, 32, 32, 32)
        x = self.conv_stack(x)
        x = self.flatten(x)
        latent_vector = self.fc(x)
        return latent_vector

class Decoder3D(nn.Module):
    """
    Decodes a 19-dimensional latent vector back into a 3D SDF grid (32x32x32).
    This is the exact reverse of the new 3-layer Encoder.
    """
    def __init__(self, latent_dim=19):
        super(Decoder3D, self).__init__()
        
        # This will project the 19D vector back to the flattened 3D size
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
        
        # Transposed convolution stack
        self.deconv_stack = nn.Sequential(
            # Input: (batch_size, 256, 4, 4, 4)
            # -> (batch_size, 128, 8, 8, 8)
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (batch_size, 64, 16, 16, 16)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (batch_size, 1, 32, 32, 32)
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x shape: (batch_size, 19)
        x = self.fc(x)
        # Reshape to (batch_size, 256, 4, 4, 4)
        x = x.view(-1, 256, 4, 4, 4)
        reconstructed_sdf = self.deconv_stack(x)
        return reconstructed_sdf

class Autoencoder3D(nn.Module):
    """
    Combines the 32x32x32 Encoder and Decoder.
    """
    def __init__(self, latent_dim=19):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 32, 32, 32)
        latent_vector = self.encoder(x)
        reconstructed_sdf = self.decoder(latent_vector)
        return reconstructed_sdf

# --- Example of how to use it ---
if __name__ == "__main__":
    # Create a dummy 3D SDF grid (batch of 2) at 32x32x32
    # (batch_size, channels, depth, height, width)
    dummy_sdf = torch.randn(2, 1, 32, 32, 32)
    
    # Initialize the autoencoder
    model = Autoencoder3D(latent_dim=19)
    
    # Pass the SDF through the model
    reconstruction = model(dummy_sdf)
    
    print(f"Original SDF shape: {dummy_sdf.shape}")
    print(f"Reconstructed SDF shape: {reconstruction.shape}")
    
    # You can also test the encoder by itself
    encoder = model.encoder
    latent_vec = encoder(dummy_sdf)
    print(f"Latent vector shape: {latent_vec.shape}") # Should be (2, 19)
    print(latent_vec)