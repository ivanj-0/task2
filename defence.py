import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from black_box_attack_cma import cma_attack
from dcgan import DCGANGenerator
from utils import save_images, generate_noise
from tqdm import tqdm


class LatentDataset(Dataset):
    def __init__(self, latent_dim, num_samples):
        self.latent_dim = latent_dim
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.latent_dim)

class LatentDAE(nn.Module):
    def __init__(self, latent_dim):
        super(LatentDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_dae(latent_dim, device, epochs=75, batch_size=512, noise_std=0.3):
    model = LatentDAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset = torch.randn(50000, latent_dim)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training DAE (Epoch {epoch+1}/{epochs})", leave=False)
        for clean_latent in pbar:
            clean_latent = clean_latent.to(device)
            noisy_latent = clean_latent + noise_std * torch.randn_like(clean_latent)

            output = model(noisy_latent)
            loss = loss_fn(output, clean_latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    return model

def defend_latent_with_dae(dae_model, z_adv):
    dae_model.eval()

    if len(z_adv.shape) > 2:
        z_adv = z_adv.squeeze(-1).squeeze(-1)

    z_adv = z_adv.detach()
    with torch.no_grad():
        z_clean = dae_model(z_adv)

    # Reshape back if needed (for DCGAN)
    if len(z_clean.shape) == 2 and z_clean.shape[-1] != 1:
        z_clean = z_clean.view(1, -1, 1, 1)

    return z_clean


if __name__ == "__main__":
    generator, _ = DCGANGenerator.load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train diffusion model
    latent_dim = generator.latent_dim
    dataset = LatentDataset(latent_dim=latent_dim, num_samples=5000000)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    dae_model = train_dae(latent_dim, device, epochs=10000)

    # Attack
    z_adv, original_img, attacked_img = cma_attack(
        generator=generator,
        latent_dim=latent_dim,
        steps=1000,
        sigma=0.0001,
        device=device
    )

    # Defense
    recovered_img = defend_latent_with_dae(dae_model, z_adv).to(device)
    recovered_img = generator.generate_image(recovered_img)
    original_img = original_img.to(device)
    attacked_img = attacked_img.to(device)
    recovered_img = recovered_img.to(device)
    save_images(torch.cat([original_img, attacked_img, recovered_img], dim=0))