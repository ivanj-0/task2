import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lpips
import os
from pgan import PGANGenerator
from black_box_attack_cma import cma_es_attack
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm

class LatentDataset(Dataset):
    def __init__(self, generator, num_samples):
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = generator.model.buildNoiseData(1)[0].shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        z, _ = self.generator.model.buildNoiseData(1)
        return z.squeeze(0)

class LatentDAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim1=512, hidden_dim2=256):
        super(LatentDAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded

def train_dae(model, loader, device, epochs=75, noise_std=0.3, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    total_iterations = epochs * len(loader)
    print(f"Total training iterations: {total_iterations} ({epochs} epochs × {len(loader)} batches)")

    model.train()
    print(f"Starting DAE training for {epochs} epochs...")
    with tqdm(total=total_iterations, desc="Training DAE") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            
            for clean_latent_flat in loader:
                clean_latent_flat = clean_latent_flat.to(device)
                noisy_latent = clean_latent_flat + noise_std * torch.randn_like(clean_latent_flat)

                output = model(noisy_latent)
                loss = loss_fn(output, clean_latent_flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "loss": f"{loss.item():.6f}"
                })
    
    print("DAE Training finished.")
    return model

def defend_latent_with_dae(dae_model, z_adv, device, original_shape):
    dae_model.eval()
    z_adv = z_adv.detach().to(device)

    z_adv_flat = z_adv.view(original_shape[0], -1)
    with torch.no_grad():
        z_clean_flat = dae_model(z_adv_flat)

    z_clean = z_clean_flat.view(original_shape)

    return z_clean


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = PGANGenerator()

    latent_dim = generator.model.buildNoiseData(1)[0].shape[1]
    print(f"Latent dimension: {latent_dim}")

    DAE_EPOCHS = 500
    DAE_BATCH_SIZE = 2048
    DAE_NOISE_STD = 0.003
    DAE_NUM_SAMPLES = DAE_BATCH_SIZE * 4

    ATTACK_ITERATIONS = 80
    ATTACK_SIGMA = 0.001

    MODELS_DIR = "models/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    DEFENSE_OUTPUT_FILENAME = "defense_comparison_cma_es_gen_noise.png"
    DAE_MODEL_PATH = os.path.join(MODELS_DIR, f"latent_dae_model_{latent_dim}.pth")
    
    dae_model = LatentDAE(latent_dim)
    dae_model = dae_model.to(device)
    if os.path.exists(DAE_MODEL_PATH):
        print(f"Loading existing DAE model from {DAE_MODEL_PATH}")
        dae_model.load_state_dict(torch.load(DAE_MODEL_PATH, map_location=device))
    else:
        print("Training new DAE model...")
        dataset = LatentDataset(generator=generator, num_samples=DAE_NUM_SAMPLES)
        loader = DataLoader(dataset, batch_size=DAE_BATCH_SIZE, shuffle=True)
        dae_model = train_dae(dae_model, loader, device, epochs=DAE_EPOCHS, noise_std=DAE_NOISE_STD)
        print(f"Saving trained DAE model to {DAE_MODEL_PATH}")
        torch.save(dae_model.state_dict(), DAE_MODEL_PATH)

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()


    print("Performing CMA-ES attack...")
    z_adv, original_img, attacked_img = cma_es_attack(
        generator=generator,
        perceptual_loss_model=perceptual_loss,
        num_iterations=ATTACK_ITERATIONS,
        sigma=ATTACK_SIGMA,
        device=device
    )
    print("Attack finished.")
    adversarial_latent_shape = z_adv.shape

    print("Applying DAE defense...")
    z_recovered = defend_latent_with_dae(dae_model, z_adv, device, adversarial_latent_shape)
    print("Defense applied.")

    print("Generating recovered image...")
    recovered_img = generator.generate_image(noise=z_recovered).detach()
    print("Recovered image generated.")

    original_img_cpu = original_img.cpu()
    attacked_img_cpu = attacked_img.cpu()
    recovered_img_cpu = recovered_img.cpu()
    comparison_tensor = torch.cat([original_img_cpu, attacked_img_cpu, recovered_img_cpu], dim=0)

    print(f"Saving comparison image to {DEFENSE_OUTPUT_FILENAME}...")
    save_images(comparison_tensor, path=DEFENSE_OUTPUT_FILENAME)
    print("Done.")