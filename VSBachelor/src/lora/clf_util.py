import numpy as np
import torch
import torch.nn as nn

# LoRA Implementation
def clf(A1, A2, rank, lambda_B, lambda_C, t, tolerance=1e-4):
    
    B = np.random.randn(A1.shape[0], rank)
    
    for i in range(t):
        C1 = np.linalg.inv(B.T @ B + lambda_C * np.eye(rank)) @ B.T @ A1
        C2 = np.linalg.inv(B.T @ B + lambda_C * np.eye(rank)) @ B.T @ A2
        B = (A1 @ C1.T + A2 @ C2.T) @ np.linalg.inv(C1 @ C1.T + C2 @ C2.T + lambda_B * np.eye(rank))
        err1 = np.linalg.norm(A1 - B @ C1, 'fro') ** 2
        err2 = np.linalg.norm(A2 - B @ C2, 'fro') ** 2
        # if (i + 1) % 100 == 0:
        # print(f"Iteration {i + 1}/{t}: Reconstruction error: Model 1: {err1:.4f}, Model 2: {err2:.4f}")
        if err1 + err2 < tolerance:
            print(f"Convergence reached at iteration {i + 1}.")
            break
        
    
    return B, C1, C2


# CNN Autoencoder Implementation
class CNNAutoencoder(nn.Module):
    """
    CNN Autoencoder for MNIST images.
    Compresses 28×28 images into a low-dimensional latent space and reconstructs them.
    """
    def __init__(self, latent_dim=32):
        super(CNNAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: CNN layers
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 28, 28) #This tells us that there is only 1 color channel in the mnist images
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Flatten(),
            # Add intermediate layer
            nn.Linear(32 * 7 * 7, 128), # Think about why it is 32 * 7 * 7 (hint: How many kernels are there in the last conv layer?)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Decoder: Transpose CNN layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32 * 7 * 7), # Hint: We are basically constructing the inverse of the encoder
            nn.ReLU(),
            nn.BatchNorm1d(32 * 7 * 7),
            nn.Unflatten(1, (32, 7, 7)),
            # Upsample: 7x7 -> 14x14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # Upsample: 14x14 -> 28x28
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """Extract embeddings from input images."""
        return self.encoder(x)
    
    def decode(self, z):
        """Reconstruct images from embeddings."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode then decode."""
        # TODO: First encode the input, then decode the latent representation
        latent_embedding = self.encode(x)
        x_recon = self.decode(latent_embedding)
        return x_recon


# Training and Evaluation Functions
def train_epoch(model, loader, optimizer, criterion, device):
    """Train autoencoder for one epoch"""
    model.train()
    total_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        
        # TODO: Forward pass - get reconstruction from model
        reconstructions = model(images)
        
        # TODO: Compute loss between reconstructions and original images
        loss = criterion(reconstructions, images)
        
        # TODO: Backward pass - zero gradients, backward, and update weights. You have worked with this before, look through old exercises and assignments if you are in doubt
        # You will write these 3 lines a billion times in your lives as AI engineers, so better get used to them now ;)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, criterion, device):
    """Evaluate autoencoder on a dataset"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # TODO: Forward pass - get reconstruction from model
            reconstructions = model(images)
            
            # TODO: Compute loss between reconstructions and original images
            loss = criterion(reconstructions, images)
            
            # Track metrics
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    """Complete training loop for autoencoder"""
    model = model.to(device)
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
    
    return model

def extract_embeddings(model, loader, device):
    """Extract embeddings and labels from a dataset"""
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # Get embeddings (encoder only, no classifier)
            emb = model.encode(images)
            
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)
    
    # Concatenate all batches
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    return embeddings, labels

