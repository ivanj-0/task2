import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
import numpy as np
import torch
import lpips
import os
from pgan import PGANGenerator
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torcheval.metrics import FrechetInceptionDistance

def calculate_fid_torcheval(images1, images2, device, target_size=(299, 299)):
    print("Calculating FID using torcheval...")
    fid_metric = FrechetInceptionDistance(feature_dim=2048, device=device)

    def preprocess_for_fid(batch):
        batch_float = batch.to(dtype=torch.float32)
        batch_resized = F.interpolate(batch_float, size=target_size, mode='bilinear', align_corners=False)

        min_val = batch_resized.min()
        max_val = batch_resized.max()

        if min_val < -0.1:
            print("Preprocessing FID: Detected range [-1, 1], shifting to [0, 1].")
            batch_shifted = (batch_resized + 1.0) / 2.0
        elif max_val > 1.1:
            print("Preprocessing FID: Detected range [0, 255], scaling to [0, 1].")
            batch_shifted = batch_resized / 255.0
        else:
            print("Preprocessing FID: Assuming input range is already [0, 1].")
            batch_shifted = batch_resized

        batch_final = torch.clamp(batch_shifted, 0.0, 1.0)

        return batch_final.to(dtype=torch.float32)

    images1_processed = preprocess_for_fid(images1)
    images2_processed = preprocess_for_fid(images2)

    fid_metric.update(images1_processed, is_real=True)
    fid_metric.update(images2_processed, is_real=False)

    fid_value = fid_metric.compute()
    print("FID calculation complete.")
    return fid_value.item()


def black_box_attack(generator, perceptual_loss_model, num_iterations, step_size, device, num_perturbations=10):
    z, _ = generator.model.buildNoiseData(1)
    z = z.to(device)
    z.requires_grad_(False)
    original_image = generator.generate_image(noise=z).detach().to(device)

    history = []
    pbar = tqdm(range(num_iterations), desc=f"Attack iters={num_iterations}, step={step_size}, perturbs={num_perturbations}", leave=False)
    current_z = z.clone().detach()

    for i in pbar:
        perturbations, _ = generator.model.buildNoiseData(num_perturbations)
        perturbations = perturbations.to(device)

        with torch.no_grad():
            batch_z = current_z.repeat(num_perturbations, 1) + step_size * perturbations
            perturbed_images = generator.generate_image(noise=batch_z).detach().to(device)
            original_batch = original_image.repeat(num_perturbations, 1, 1, 1)
            dists = perceptual_loss_model(original_batch, perturbed_images).squeeze()

        scores = dists.cpu().numpy()
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        history.append(best_score)

        current_z = current_z + step_size * perturbations[best_idx:best_idx+1].detach()
        pbar.set_postfix({"LPIPS": f"{best_score:.4f}"})

    attacked_image = generator.generate_image(noise=current_z).detach().to(device)
    return current_z, original_image, attacked_image, history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {'iterations': 100, 'step': 0.01, 'perts': 20}
    config_key = f"iters{config['iterations']}_step{config['step']}_perts{config['perts']}"
    print(f"Running with fixed config: {config_key}")

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    generator = PGANGenerator()
    try:
        generator.model.to(device)
        generator.model.eval()
        for param in generator.model.parameters():
            param.requires_grad = False
    except AttributeError:
        print("Could not set generator model to eval mode or disable gradients.")
    except Exception as e:
        print(f"An error occurred moving generator to device or setting mode: {e}")
        pass
    print("Generator loaded and moved to device.")

    num_attack_runs = 100
    attack_success_threshold = 0.1
    base_save_dir = os.path.join("runs", "basic_attack", "pgan")
    os.makedirs(base_save_dir, exist_ok=True)

    all_original_imgs = []
    all_attacked_imgs = []
    successful_attacks = 0
    final_lpips_scores = []

    print(f"\n--- Starting {num_attack_runs} Attack Runs ---")
    for run_i in tqdm(range(num_attack_runs), desc="Attack Runs"):
        z_final, original_img, attacked_img, history = black_box_attack(
            generator, perceptual_loss,
            num_iterations=config['iterations'],
            step_size=config['step'],
            device=device,
            num_perturbations=config['perts']
        )

        all_original_imgs.append(original_img.cpu())
        all_attacked_imgs.append(attacked_img.cpu())

        with torch.no_grad():
            dist = perceptual_loss(original_img.to(device), attacked_img.to(device)).item()
        final_lpips_scores.append(dist)

        if dist > attack_success_threshold:
            successful_attacks += 1


        comp = torch.cat([original_img.cpu(), attacked_img.cpu()], dim=0)
        fname = os.path.join(base_save_dir, f"run_{run_i+1}.png")
        save_images(comp, path=fname)

    print(f"\n--- Results for Config: {config_key} ({num_attack_runs} runs) ---")

    success_rate = (successful_attacks / num_attack_runs) * 100
    print(f"Attack Success Rate (LPIPS > {attack_success_threshold}): {success_rate:.2f}% ({successful_attacks}/{num_attack_runs})")

    original_batch = torch.cat(all_original_imgs, dim=0).to(device)
    attacked_batch = torch.cat(all_attacked_imgs, dim=0).to(device)

    print("\n--- LPIPS Evaluation (All Runs) ---")
    with torch.no_grad():
        all_run_dists = perceptual_loss(original_batch, attacked_batch).squeeze()

    if all_run_dists.dim() == 0:
         print(f"Average LPIPS (original vs attacked): {all_run_dists.item():.4f}")
    elif all_run_dists.dim() == 1:
         print(f"Average LPIPS (original vs attacked): {all_run_dists.mean().item():.4f}")
         print(f"Std Dev LPIPS: {all_run_dists.std().item():.4f}")
    else:
         print(f"Warning: Unexpected shape for LPIPS distances: {all_run_dists.shape}.")
         print(f"Individual LPIPS scores: {all_run_dists.cpu().numpy()}")


    print("\n--- Fréchet Inception Distance (FID) score ---")
    try:
        fid_value = calculate_fid_torcheval(original_batch.to(device), attacked_batch.to(device), device)
        print(f"FID (original vs attacked): {fid_value:.4f}")
    except ImportError:
         print("Could not calculate FID: torcheval not found. Please install it (`pip install torcheval`).")
    except Exception as e:
         print(f"Could not calculate FID: {e}")


    print("\n--- Attack Script Finished ---")