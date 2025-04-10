import numpy as np
import torch
import lpips
import os
from pgan import PGANGenerator
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm


def black_box_attack(generator, perceptual_loss_model, num_iterations, step_size, device, num_perturbations=10):
    """
    Performs a simple black-box optimization attack using batched LPIPS calculation.
    Tries to maximize the LPIPS distance from the original image at each step.
    """
    # Initialize latent vector randomly on the correct device
    z, _ = generator.model.buildNoiseData(1)
    z = z.to(device)
    z.requires_grad_(False)

    # Generate original image from target GAN
    original_image = generator.generate_image(noise=z).detach()

    pbar = tqdm(range(num_iterations), desc="Basic Attack")
    for i in pbar:
        # Sample neighboring latent vectors (perturbations)
        perturbations, _ = generator.model.buildNoiseData(num_perturbations)
        perturbations = perturbations.to(device)

        # Create batch of perturbed latent vectors
        perturbed_z_batch = z.repeat(num_perturbations, 1) + step_size * perturbations

        # Generate batch of perturbed images
        perturbed_images_batch = generator.generate_image(noise=perturbed_z_batch).detach()

        # Prepare original image batch for comparison
        original_image_batch = original_image.repeat(num_perturbations, 1, 1, 1)

        # Calculate perceptual distances (LPIPS) for the entire batch
        with torch.no_grad():
            dists = perceptual_loss_model(original_image_batch.to(device), perturbed_images_batch.to(device)).squeeze()
        
        scores = dists.cpu().numpy()

        # Select perturbation direction that MAXIMIZES LPIPS score
        best_idx = np.argmax(scores)
        best_direction = perturbations[best_idx:best_idx+1] # Keep batch dimension

        z = z + step_size * best_direction
        
        pbar.set_postfix({
            "Best LPIPS Score": f"{scores[best_idx]:.4f}"
        })

    print("Optimization finished.")
    attacked_image = generator.generate_image(noise=z).detach()

    return z, original_image, attacked_image

if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    # Load the generator model
    generator = DCGANGenerator()
    print("Generator loaded.")

    # --- Configuration ---
    NUM_ITERATIONS = 100
    STEP_SIZE = 0.01
    NUM_PERTURBATIONS = 20
    OUTPUT_FILENAME = "attack_comparison_batched.png"
    # ---------------------

    # Run the attack
    z_final, original_image, attacked_image = black_box_attack(
        generator,
        perceptual_loss,
        num_iterations=NUM_ITERATIONS,
        step_size=STEP_SIZE,
        device=device,
        num_perturbations=NUM_PERTURBATIONS
    )

    # Save the original and attacked images side-by-side
    comparison_tensor = torch.cat([original_image.cpu(), attacked_image.cpu()], dim=0)
    print(f"Saving comparison image to {OUTPUT_FILENAME}...")
    save_images(comparison_tensor, path=OUTPUT_FILENAME)
    print("Done.")