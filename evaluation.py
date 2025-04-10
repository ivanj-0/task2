import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from black_box_attack_cma import cma_attack
from dcgan import DCGANGenerator
from defence import LatentDataset, train_dae, defend_latent_with_dae
from utils import save_images, calculate_fid
import lpips

# Create the directory if it doesn't exist
os.makedirs("models/", exist_ok=True)

samples = 100
original_imgs = []
attacked_imgs = []
recovered_imgs = []

for i in tqdm(range(samples)):
    generator, device = DCGANGenerator.load_model()

    # Setup for DAE model
    latent_dim = generator.latent_dim
    dae_model_path = f"models/dae_model_{latent_dim}.pt"
    
    # Check if the model already exists
    if os.path.exists(dae_model_path):
        print(f"Loading existing DAE model from {dae_model_path}")
        dae_model = torch.load(dae_model_path)
        dae_model.to(device)
    else:
        print(f"Training new DAE model...")
        # Create dataset and dataloader
        dataset = LatentDataset(latent_dim=latent_dim, num_samples=5000000)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Train the model
        dae_model = train_dae(latent_dim, device, epochs=10000)
        
        # Save the trained model
        print(f"Saving DAE model to {dae_model_path}")
        torch.save(dae_model, dae_model_path)

    # Attack
    z_adv, original_img, attacked_img = cma_attack(
        generator=generator,
        latent_dim=latent_dim,
        steps=200,
        sigma=0.0001,
        device=device
    )

    # Defense
    recovered_img = defend_latent_with_dae(dae_model, z_adv).to(device)
    recovered_img = generator.generate_image(recovered_img)
    original_img = original_img.to(device)
    attacked_img = attacked_img.to(device)
    recovered_img = recovered_img.to(device)
    save_images(torch.cat([original_img, attacked_img, recovered_img], dim=0), f"dcgan/{i}")
    # Store images for evaluation
    original_imgs.append(original_img)
    attacked_imgs.append(attacked_img)
    recovered_imgs.append(recovered_img)

original_batch = torch.cat(original_imgs, dim=0)
attacked_batch = torch.cat(attacked_imgs, dim=0)
recovered_batch = torch.cat(recovered_imgs, dim=0)

# Calculate LPIPS between the different image sets
perceptual_loss = lpips.LPIPS(net='vgg').to(device)
perceptual_loss.eval()

with torch.no_grad():
    # Calculate distances between original and attacked
    attack_dists = perceptual_loss(original_batch, attacked_batch).squeeze()

    # Calculate distances between original and recovered
    recovered_dists = perceptual_loss(original_batch, recovered_batch).squeeze()

    # Calculate distances between attacked and recovered
    attack_recovered_dists = perceptual_loss(attacked_batch, recovered_batch).squeeze()

# Calculate and print average distances
print(f"Average LPIPS (original vs attacked): {attack_dists.mean().item():.4f}")
print(f"Average LPIPS (original vs recovered): {recovered_dists.mean().item():.4f}")
print(f"Average LPIPS (attacked vs recovered): {attack_recovered_dists.mean().item():.4f}")


# Calculate FID scores
print("\n--- Fréchet Inception Distance (FID) scores ---")
fid_orig_attacked = calculate_fid(original_batch, attacked_batch, device)
fid_orig_recovered = calculate_fid(original_batch, recovered_batch, device)
fid_attacked_recovered = calculate_fid(attacked_batch, recovered_batch, device)

print(f"FID (original vs attacked): {fid_orig_attacked:.4f}")
print(f"FID (original vs recovered): {fid_orig_recovered:.4f}")
print(f"FID (attacked vs recovered): {fid_attacked_recovered:.4f}")



