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
from sklearn.decomposition import PCA

class DefenseAgainstBlackBoxAttack:
    def __init__(self, generator, device):
        self.generator = generator
        self.device = device
        
    def moving_average_filter(self, latent_z, window_size=5):
        z = latent_z.view(1, 1, -1)
        padding = window_size // 2
        filtered_z = F.avg_pool1d(z, kernel_size=window_size, stride=1, padding=padding)
        return filtered_z.view_as(latent_z)
    
    def clip_noise(self, latent_z, threshold=2.0):
        return torch.clamp(latent_z, -threshold, threshold)
    
    def pca_denoise(self, latent_z, n_components=10000, training_samples=10000000000):
        training_vectors = []
        for _ in range(training_samples):
            z, _ = self.generator.model.buildNoiseData(1)
            training_vectors.append(z.cpu().numpy().flatten())
        training_matrix = np.vstack(training_vectors)
        pca = PCA(n_components=n_components)
        pca.fit(training_matrix)
        z_np = latent_z.cpu().numpy().reshape(1, -1)
        transformed = pca.transform(z_np)
        reconstructed = pca.inverse_transform(transformed)
        return torch.tensor(reconstructed, device=self.device).view_as(latent_z)
    
    def defend(self, latent_z, methods=None):
        if methods is None:
            methods = ['moving_average', 'clip']
            
        defended_z = latent_z.clone()
        
        for method in methods:
            if method == 'moving_average':
                defended_z = self.moving_average_filter(defended_z)
            elif method == 'clip':
                defended_z = self.clip_noise(defended_z)
            elif method == 'pca':
                defended_z = self.pca_denoise(defended_z)
                
        return defended_z


if __name__ == "__main__":
    ATTACK_ITERATIONS = 100
    ATTACK_SIGMA = 0.01
    SAVE_PATH = "results/black_box_defense"
    DEFENSE_METHODS = ['clip', 'moving_average']
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = PGANGenerator()

    latent_dim = generator.model.buildNoiseData(1)[0].shape[1]
    print(f"Latent dimension: {latent_dim}")
    
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
    
    print("Applying defense mechanisms...")
    defense = DefenseAgainstBlackBoxAttack(generator, device)
    
    print(f"Shape of adversarial latent vector: {z_adv.shape}")
    
    defended_z = defense.defend(z_adv, methods=DEFENSE_METHODS)
    
    with torch.no_grad():
        defended_img = generator.generate_image(noise=defended_z)
    
    print("Saving images...")
    save_images(original_img, os.path.join(SAVE_PATH, "original.png"))
    save_images(attacked_img, os.path.join(SAVE_PATH, "attacked.png"))
    save_images(defended_img, os.path.join(SAVE_PATH, "defended.png"))
    
    with torch.no_grad():
        original_img_device = original_img.to(device)
        attacked_img_device = attacked_img.to(device)
        defended_img_device = defended_img.to(device)
        
        orig_vs_attacked_dist = perceptual_loss(original_img_device, attacked_img_device).item()
        orig_vs_defended_dist = perceptual_loss(original_img_device, defended_img_device).item()
    
    print(f"Perceptual distance - Original vs Attacked: {orig_vs_attacked_dist:.4f}")
    print(f"Perceptual distance - Original vs Defended: {orig_vs_defended_dist:.4f}")
    print(f"Defense effectiveness: {(orig_vs_attacked_dist - orig_vs_defended_dist) / orig_vs_attacked_dist * 100:.2f}%")
    
    print(f"All images saved to {SAVE_PATH}")