import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lpips
import os
import numpy as np
from pgan import PGANGenerator
from black_box_attack_cma import cma_es_attack
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm
import matplotlib.pyplot as plt
from defence_cma import improved_latent_defense_cma

class LatentDataset(Dataset):
    def __init__(self, generator, num_samples, perturbation_range=(0.001, 0.005), device="cpu"):
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = generator.model.buildNoiseData(1)[0].shape[1]
        self.perturbation_range = perturbation_range
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        z_original, _ = self.generator.model.buildNoiseData(1)
        z_original = z_original.to(self.device)
        
        perturbation, _ = self.generator.model.buildNoiseData(1)
        perturbation = perturbation.to(self.device)
        
        scale = torch.rand(1, device=self.device) * (self.perturbation_range[1] - self.perturbation_range[0]) + self.perturbation_range[0]
        z_perturbed = z_original + scale * perturbation
        
        return z_perturbed.squeeze(0), z_original.squeeze(0)

class LatentEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512):
        super(LatentEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512):
        super(LatentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

class LatentDenoisingGAN:
    def __init__(self, latent_dim, device="cuda", hidden_dim=512, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.device = device
        
        self.encoder = LatentEncoder(latent_dim, hidden_dim).to(device)
        self.discriminator = LatentDiscriminator(latent_dim, hidden_dim).to(device)
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta1, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()

    def save(self, path):
        state_dict = {
            'encoder': self.encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }
        torch.save(state_dict, path)

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.encoder.load_state_dict(state_dict['encoder'])
        self.discriminator.load_state_dict(state_dict['discriminator'])

def train_gan(gan_model, loader, device, epochs=75, lambda_rec=10.0):
    batch_size = loader.batch_size
    
    real_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)
    
    total_iterations = epochs * len(loader)
    print(f"Total training iterations: {total_iterations} ({epochs} epochs × {len(loader)} batches)")
    
    with tqdm(total=total_iterations, desc="Training GAN") as pbar:
        for epoch in range(epochs):
            total_disc_loss = 0
            total_gen_loss = 0
            
            for perturbed_z, original_z in loader:
                perturbed_z = perturbed_z.to(device)
                original_z = original_z.to(device)
                
                current_batch_size = perturbed_z.size(0)
                if current_batch_size != batch_size:
                    real_label_batch = torch.ones(current_batch_size, 1, device=device)
                    fake_label_batch = torch.zeros(current_batch_size, 1, device=device)
                else:
                    real_label_batch = real_label
                    fake_label_batch = fake_label
                
                gan_model.discriminator_optimizer.zero_grad()
                
                real_output = gan_model.discriminator(original_z)
                real_loss = gan_model.adversarial_loss(real_output, real_label_batch)
                
                denoised_z = gan_model.encoder(perturbed_z)
                fake_output = gan_model.discriminator(denoised_z.detach())
                fake_loss = gan_model.adversarial_loss(fake_output, fake_label_batch)
                
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                gan_model.discriminator_optimizer.step()
                
                total_disc_loss += disc_loss.item()
                
                gan_model.encoder_optimizer.zero_grad()
                
                fake_output = gan_model.discriminator(denoised_z)
                gen_adv_loss = gan_model.adversarial_loss(fake_output, real_label_batch)
                
                rec_loss = gan_model.reconstruction_loss(denoised_z, original_z)
                
                gen_loss = gen_adv_loss + lambda_rec * rec_loss
                gen_loss.backward()
                gan_model.encoder_optimizer.step()
                
                total_gen_loss += gen_loss.item()
                
                pbar.update(1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "D_loss": f"{disc_loss.item():.6f}",
                    "G_loss": f"{gen_loss.item():.6f}"
                })
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D_loss: {total_disc_loss/len(loader):.6f}, "
                      f"G_loss: {total_gen_loss/len(loader):.6f}")
    
    print("GAN Training finished.")
    return gan_model

def defend_latent_with_gan(gan_model, z_adv, device, original_shape):
    """Cleans an adversarial latent vector using the trained GAN encoder."""
    gan_model.encoder.eval()
    z_adv = z_adv.detach().to(device)
    
    batch_size = original_shape[0]
    z_adv_flat = z_adv.view(batch_size, -1)
    
    with torch.no_grad():
        z_clean_flat = gan_model.encoder(z_adv_flat)
    
    z_clean = z_clean_flat.view(original_shape)
    
    return z_clean

def compute_lpips_safely(img1, img2, perceptual_loss_model, device):
    """Compute LPIPS distance ensuring all tensors are on the same device"""
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    perceptual_loss_model = perceptual_loss_model.to(device)
    
    with torch.no_grad():
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        lpips_value = perceptual_loss_model(img1, img2)
        
        if lpips_value.numel() > 1:
            lpips_value = lpips_value.mean()
            
        return lpips_value.item()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = DCGANGenerator()

    latent_dim = generator.model.buildNoiseData(1)[0].shape[1]
    print(f"Latent dimension: {latent_dim}")

    GAN_EPOCHS = 100
    GAN_BATCH_SIZE = 2048
    GAN_PERTURBATION_RANGE = (0.01, 0.005)
    GAN_NUM_SAMPLES = GAN_BATCH_SIZE * 10
    GAN_LAMBDA_REC = 10.0

    ATTACK_ITERATIONS = 100
    ATTACK_SIGMA = 0.01

    LATENT_DEFENSE_STEPS = 100
    LATENT_DEFENSE_POP_SIZE = 32
    LATENT_DEFENSE_SIGMA0 = 0.01
    LATENT_DEFENSE_WEIGHT_LPIPS = 0.5
    LATENT_DEFENSE_WEIGHT_L2 = 0.5

    MODELS_DIR = "models/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    DEFENSE_OUTPUT_FILENAME = "defense_comparison_combined.png"
    GAN_MODEL_PATH = os.path.join(MODELS_DIR, f"latent_gan_model_{latent_dim}.pth")
    LPIPS_PLOT_FILENAME = "defense_lpips_history.png"

    gan_model = LatentDenoisingGAN(latent_dim=latent_dim, device=device)

    if os.path.exists(GAN_MODEL_PATH):
        print(f"Loading existing GAN model from {GAN_MODEL_PATH}")
        gan_model.load(GAN_MODEL_PATH, device)
    else:
        print("Training new GAN model...")

        dataset = LatentDataset(
            generator=generator, 
            num_samples=GAN_NUM_SAMPLES,
            perturbation_range=GAN_PERTURBATION_RANGE,
            device="cpu"
        )
        loader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, drop_last=False)
        

        gan_model = train_gan(
            gan_model=gan_model, 
            loader=loader, 
            device=device, 
            epochs=GAN_EPOCHS, 
            lambda_rec=GAN_LAMBDA_REC
        )
        
        print(f"Saving trained GAN model to {GAN_MODEL_PATH}")
        gan_model.save(GAN_MODEL_PATH)


    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    z_adv, original_img, attacked_img = cma_es_attack(
        generator=generator,
        perceptual_loss_model=perceptual_loss,
        num_iterations=ATTACK_ITERATIONS,
        sigma=ATTACK_SIGMA,
        device=device
    )
    adversarial_latent_shape = z_adv.shape

    z_recovered_gan = defend_latent_with_gan(gan_model, z_adv, device, adversarial_latent_shape)

    recovered_img_combined, z_recovered_combined, defense_lpips_history = improved_latent_defense_cma(
        generator=generator,
        original_img=original_img,
        z_init=z_recovered_gan,
        latent_dim=latent_dim,
        population_size=LATENT_DEFENSE_POP_SIZE,
        max_iterations=LATENT_DEFENSE_STEPS,
        sigma0=LATENT_DEFENSE_SIGMA0,
        weight_lpips=LATENT_DEFENSE_WEIGHT_LPIPS,
        weight_l2=LATENT_DEFENSE_WEIGHT_L2,
        device=device
    )

    attack_lpips = compute_lpips_safely(original_img, attacked_img, perceptual_loss, device)
    combined_defense_lpips = compute_lpips_safely(original_img, recovered_img_combined, perceptual_loss, device)
    
    print(f"Attack LPIPS: {attack_lpips:.6f}")
    print(f"Defense LPIPS: {combined_defense_lpips:.6f}")
    
    original_img_cpu = original_img.detach().cpu()
    attacked_img_cpu = attacked_img.detach().cpu()
    recovered_img_combined_cpu = recovered_img_combined.detach().cpu()
    comparison_tensor = torch.cat([original_img_cpu, attacked_img_cpu, recovered_img_combined_cpu], dim=0)
    save_images(comparison_tensor, path=DEFENSE_OUTPUT_FILENAME)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(defense_lpips_history)), defense_lpips_history)
    plt.title("LPIPS Score over Defense Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("LPIPS Score")
    plt.grid(True)
    plt.savefig(LPIPS_PLOT_FILENAME)

    

