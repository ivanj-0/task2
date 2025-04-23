import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
import numpy as np
import torch
import lpips
import os
import cma
from pgan import PGANGenerator
from dcgan import DCGANGenerator


from utils import save_images
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torcheval.metrics import FrechetInceptionDistance

def calculate_fid_torcheval(images1, images2, device, target_size=(299, 299)):
    """
    Calculates FID using torcheval.metrics.FrechetInceptionDistance.
    Ensures input images are preprocessed to float32 and range [0, 1].
    """
    print("Calculating FID using torcheval...")
    fid_metric = FrechetInceptionDistance(feature_dim=2048, device=device)

    def preprocess_for_fid(batch):
        batch_float = batch.to(dtype=torch.float32)
        batch_resized = F.interpolate(batch_float, size=target_size, mode='bilinear', align_corners=False)

        min_val = batch_resized.min()
        max_val = batch_resized.max()


        if min_val < -0.1 and max_val > 0.1:
            print("Preprocessing FID: Detected range likely [-1, 1], shifting to [0, 1].")
            batch_shifted = (batch_resized + 1.0) / 2.0
        elif max_val > 1.1:
            print("Preprocessing FID: Detected range likely [0, 255], scaling to [0, 1].")
            batch_shifted = batch_resized / 255.0
        elif min_val >= -0.1 and max_val <= 1.1:
             print("Preprocessing FID: Assuming input range is already close to [0, 1]. Clamping.")
             batch_shifted = batch_resized
        else:
            print(f"Preprocessing FID: Unsure about input range (min: {min_val}, max: {max_val}). Assuming [0, 1].")
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



def cma_es_attack(generator, perceptual_loss_model, num_iterations, sigma, device):
    z, _ = generator.model.buildNoiseData(1)
    z = z.to(device)
    z.requires_grad_(False)

    original_image = generator.generate_image(noise=z).detach()


    z_np = z.cpu().numpy().reshape(-1)
    es = cma.CMAEvolutionStrategy(z_np, sigma)

    print("Starting CMA-ES optimization...")
    best_overall_score = -np.inf
    best_z_tensor = z.clone()

    pbar = tqdm(range(num_iterations), desc="CMA Attack")
    for i in pbar:

        solutions_np = es.ask()
        num_solutions = len(solutions_np)


        solutions_list = []
        for sol_np in solutions_np:


            z_tensor = torch.tensor(sol_np, dtype=torch.float32, device=device).view(1, -1)
            solutions_list.append(z_tensor)

        perturbed_z_batch = torch.cat(solutions_list, dim=0)


        perturbed_images_batch = generator.generate_image(noise=perturbed_z_batch).detach()


        original_image_batch = original_image.repeat(num_solutions, 1, 1, 1)


        with torch.no_grad():

            dists = perceptual_loss_model(original_image_batch.to(device), perturbed_images_batch.to(device)).squeeze()


        scores_for_cma = -dists.cpu().numpy()
        scores_for_reporting = dists.cpu().numpy()


        es.tell(solutions_np, scores_for_cma)


        current_best_idx = np.argmax(scores_for_reporting)
        current_best_score = scores_for_reporting[current_best_idx]


        if current_best_score > best_overall_score:
             best_overall_score = current_best_score

             best_z_tensor = solutions_list[current_best_idx].detach().clone()

        pbar.set_postfix({
            "Best Batch LPIPS": f"{current_best_score:.4f}",
            "Overall Best LPIPS": f"{best_overall_score:.4f}"
        })

    print("Optimization finished.")

    final_z = best_z_tensor
    attacked_image = generator.generate_image(noise=final_z).detach()


    return final_z, original_image, attacked_image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    config = {'iterations': 100, 'sigma': 0.01}
    config_key = f"cma_iters{config['iterations']}_sigma{config['sigma']}"
    print(f"Running with CMA-ES config: {config_key}")

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    generator = DCGANGenerator()
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
    base_save_dir = os.path.join("runs", "cma_attack", "dcgan")
    os.makedirs(base_save_dir, exist_ok=True)

    all_original_imgs = []
    all_attacked_imgs = []
    successful_attacks = 0
    final_lpips_scores = []

    print(f"\n--- Starting {num_attack_runs} CMA Attack Runs ---")
    for run_i in tqdm(range(num_attack_runs), desc="CMA Attack Runs"):


        z_final, original_img, attacked_img = cma_es_attack(
            generator, perceptual_loss,
            num_iterations=config['iterations'],
            sigma=config['sigma'],
            device=device
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

    print(f"\n--- Results for CMA Config: {config_key} ({num_attack_runs} runs) ---")


    success_rate = (successful_attacks / num_attack_runs) * 100
    print(f"Attack Success Rate (LPIPS > {attack_success_threshold}): {success_rate:.2f}% ({successful_attacks}/{num_attack_runs})")


    original_batch = torch.cat(all_original_imgs, dim=0).to(device)
    attacked_batch = torch.cat(all_attacked_imgs, dim=0).to(device)


    print("\n--- LPIPS Evaluation (All Runs) ---")
    if final_lpips_scores:
        avg_lpips = np.mean(final_lpips_scores)
        std_lpips = np.std(final_lpips_scores)
        print(f"Average Final LPIPS (original vs attacked): {avg_lpips:.4f}")
        print(f"Std Dev Final LPIPS: {std_lpips:.4f}")
    else:
        print("No LPIPS scores recorded.")


    print("\n--- Fréchet Inception Distance (FID) score ---")
    if not all_original_imgs or not all_attacked_imgs:
         print("Skipping FID calculation: No images generated.")
    else:
        try:

            fid_value = calculate_fid_torcheval(original_batch.to(device), attacked_batch.to(device), device)
            print(f"FID (original vs attacked): {fid_value:.4f}")
        except ImportError:
             print("Could not calculate FID: torcheval not found. Please install it (`pip install torcheval`).")
        except Exception as e:
             print(f"Could not calculate FID: {e}")


    print("\n--- CMA Attack Script Finished ---")