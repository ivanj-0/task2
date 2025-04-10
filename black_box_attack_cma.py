import numpy as np
import torch
import lpips
import cma
import os
from pgan import PGANGenerator
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm


def cma_es_attack(generator, perceptual_loss_model, num_iterations, sigma, device):
    """
    Performs a black-box optimization attack using CMA-ES.
    Tries to maximize the LPIPS distance from the original image at each step.
    """
    z, _ = generator.model.buildNoiseData(1)
    z = z.to(device)
    z.requires_grad_(False)

    original_image = generator.generate_image(noise=z).detach()

    # Initialize CMA-ES
    z_np = z.cpu().numpy().reshape(-1)
    es = cma.CMAEvolutionStrategy(z_np, sigma)

    print("Starting CMA-ES optimization...")
    best_overall_score = -np.inf
    best_z = z.clone()

    pbar = tqdm(range(num_iterations), desc="CMA Attack")
    for i in pbar:
        # Ask CMA-ES for candidate solutions (latent vectors)
        solutions_np = es.ask()
        num_solutions = len(solutions_np)

        solutions_list = []
        for sol_np in solutions_np:
            z_tensor = torch.tensor(sol_np, dtype=torch.float32, device=device).unsqueeze(0)
            solutions_list.append(z_tensor)
        
        perturbed_z_batch = torch.cat(solutions_list, dim=0)
        perturbed_images_batch = generator.generate_image(noise=perturbed_z_batch).detach()

        original_image_batch = original_image.repeat(num_solutions, 1, 1, 1)

        with torch.no_grad():
            dists = perceptual_loss_model(original_image_batch.to(device), perturbed_images_batch.to(device)).squeeze()

        scores = dists.cpu().numpy()

        es.tell(solutions_np, -scores)

        current_best_idx = np.argmax(scores)
        current_best_score = scores[current_best_idx]

        if current_best_score > best_overall_score:
             best_overall_score = current_best_score
             best_z = solutions_list[current_best_idx].detach().clone()

        pbar.set_postfix({
            "Best Batch LPIPS": f"{current_best_score:.4f}",
            "Overall Best LPIPS": f"{best_overall_score:.4f}"
        })

    print("Optimization finished.")
    final_z = torch.tensor(best_z, dtype=torch.float32, device=device)
    attacked_image = generator.generate_image(noise=final_z).detach()

    return final_z, original_image, attacked_image

if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the perceptual loss model
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    generator = PGANGenerator()
    print("Generator loaded.")

    # --- Configuration ---
    NUM_ITERATIONS = 1000
    SIGMA = 0.001
    OUTPUT_FILENAME = "attack_comparison_cma_es.png"
    # ---------------------

    # Run the attack
    z_final, original_image, attacked_image = cma_es_attack(
        generator=generator,
        perceptual_loss_model=perceptual_loss,
        num_iterations=NUM_ITERATIONS,
        sigma=SIGMA,
        device=device
    )

    comparison_tensor = torch.cat([original_image.cpu(), attacked_image.cpu()], dim=0)
    print(f"Saving comparison image to {OUTPUT_FILENAME}...")
    save_images(comparison_tensor, path=OUTPUT_FILENAME)
    print("Done.")