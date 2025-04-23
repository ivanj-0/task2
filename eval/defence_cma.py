import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
import torch
import torch.nn.functional as F
import lpips
import os
import numpy as np
from black_box_attack_cma import cma_es_attack
from dcgan import DCGANGenerator
from pgan import PGANGenerator
from utils import save_images
import cma
import matplotlib.pyplot as plt
from tqdm import tqdm
from torcheval.metrics import FrechetInceptionDistance


def improved_latent_defense_cma(
    generator,
    original_img,
    z_init,
    latent_dim,
    population_size=16,
    max_iterations=500,
    sigma0=0.5,
    weight_lpips=0.5,
    weight_l2=0.5,
    device=None,
    verbose=True
):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_img = original_img.to(device).detach()

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    z_init_np = z_init.detach().cpu().numpy().flatten()

    lpips_history = []

    es = cma.CMAEvolutionStrategy(
        z_init_np,
        sigma0,
        {'popsize': population_size}
    )

    iteration = 0
    while not es.stop() and iteration < max_iterations:
        solutions = es.ask()

        fitnesses = []
        lpips_losses_gen = []
        for z_vector in solutions:
            z_reshaped = torch.tensor(z_vector.reshape(z_init.shape),
                                    device=device,
                                    dtype=torch.float32)
            with torch.no_grad():
                gen_img = generator.generate_image(z_reshaped)
                gen_img = gen_img.to(device)
                lpips_loss_val = perceptual_loss(original_img, gen_img).mean().item()
                l2_loss_val = torch.nn.functional.mse_loss(gen_img, original_img).item()
                total_loss = weight_lpips * lpips_loss_val + weight_l2 * l2_loss_val

            fitnesses.append(total_loss)
            lpips_losses_gen.append(lpips_loss_val)

        es.tell(solutions, fitnesses)

        best_gen_idx = np.argmin(fitnesses)
        best_gen_lpips = lpips_losses_gen[best_gen_idx]
        lpips_history.append(best_gen_lpips)

        best_gen_fitness = fitnesses[best_gen_idx]

        if verbose and iteration % 10 == 0:
            print(f"Generation {iteration}, best fitness: {best_gen_fitness:.6f}, best LPIPS: {best_gen_lpips:.6f}")

        iteration += 1

    best_solution = es.result[0]
    best_fitness = es.result[1]

    if verbose:
        print(f"Final best fitness: {best_fitness:.6f}")

    best_z = torch.tensor(best_solution.reshape(z_init.shape),
                          device=device,
                          dtype=torch.float32)

    with torch.no_grad():
        best_img = generator.generate_image(best_z)

    if verbose:
        print(f"CMA-ES optimization complete.")

    return best_img, best_z, lpips_history

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

if __name__ == "__main__":
    
    output_dir = "runs/cma_defence/pgan"
    os.makedirs(output_dir, exist_ok=True)
    
    
    NUM_RUNS = 100
    LPIPS_THRESHOLD = 0.1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        generator = PGANGenerator()
    except NameError:
        print("Error: DCGANGenerator not defined. Please ensure it's imported correctly.")
        exit()
    except Exception as e:
        print(f"Error loading generator: {e}")
        exit()

    try:
        dummy_noise = generator.model.buildNoiseData(1)[0]
        latent_dim = dummy_noise.shape[1]
        print(f"Latent dimension: {latent_dim}")
    except AttributeError:
        print("Error: Cannot determine latent_dim. 'generator.model' might not have 'buildNoiseData' or shape is unexpected.")
        latent_dim = 512
        print(f"Warning: Using default latent_dim: {latent_dim}")
    except Exception as e:
        print(f"Error determining latent_dim: {e}")
        exit()

    
    ATTACK_ITERATIONS = 100
    ATTACK_SIGMA = 0.01

    LATENT_DEFENSE_STEPS = 100
    LATENT_DEFENSE_POP_SIZE = 32
    LATENT_DEFENSE_SIGMA0 = 0.01
    LATENT_DEFENSE_WEIGHT_LPIPS = 0.5
    LATENT_DEFENSE_WEIGHT_L2 = 0.5

    
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()
    
    
    all_original_imgs = []
    all_attacked_imgs = []
    all_recovered_imgs = []
    attack_lpips_scores = []
    defense_lpips_scores = []
    final_lpips_scores = []
    attack_success_count = 0
    defense_success_count = 0
    
    
    print(f"Running {NUM_RUNS} iterations of attack and defense...")
    for run in tqdm(range(1, NUM_RUNS + 1)):
        try:
            
            z_adv, original_img, attacked_img = cma_es_attack(
                generator=generator,
                perceptual_loss_model=perceptual_loss,
                num_iterations=ATTACK_ITERATIONS,
                sigma=ATTACK_SIGMA,
                device=device,
            )
            
            
            recovered_img, z_recovered, defense_lpips_history = improved_latent_defense_cma(
                generator=generator,
                original_img=original_img,
                z_init=z_adv,
                latent_dim=latent_dim,
                population_size=LATENT_DEFENSE_POP_SIZE,
                max_iterations=LATENT_DEFENSE_STEPS,
                sigma0=LATENT_DEFENSE_SIGMA0,
                weight_lpips=LATENT_DEFENSE_WEIGHT_LPIPS,
                weight_l2=LATENT_DEFENSE_WEIGHT_L2,
                device=device,
                verbose=False if run > 1 else True
            )
            
            
            with torch.no_grad():
                original_img = original_img.to(device)
                attacked_img = attacked_img.to(device)
                recovered_img = recovered_img.to(device)
                
                
                lpips_attacked = perceptual_loss(original_img, attacked_img).mean().item()
                lpips_recovered = perceptual_loss(original_img, recovered_img).mean().item()
                
                
                if lpips_attacked > LPIPS_THRESHOLD:
                    attack_success_count += 1
                
                if lpips_recovered < LPIPS_THRESHOLD:
                    defense_success_count += 1
                
                
                attack_lpips_scores.append(lpips_attacked)
                defense_lpips_scores.append(lpips_recovered)
                final_lpips_scores.append(lpips_recovered)
            
            
            all_original_imgs.append(original_img.cpu())
            all_attacked_imgs.append(attacked_img.cpu())
            all_recovered_imgs.append(recovered_img.cpu())
            
            
            original_img_cpu = original_img.cpu()
            attacked_img_cpu = attacked_img.cpu()
            recovered_img_cpu = recovered_img.cpu()
            
            comparison_img = torch.cat([original_img_cpu, attacked_img_cpu, recovered_img_cpu], dim=0)
            save_path = os.path.join(output_dir, f"run_{run}.png")
            save_images(comparison_img, path=save_path)
            
            
            if run == 1 or run == NUM_RUNS:
                print(f"\nRun {run} Metrics:")
                print(f"LPIPS - Attacked: {lpips_attacked:.4f}, Recovered: {lpips_recovered:.4f}")
                
        except Exception as e:
            print(f"\nError in run {run}: {e}")
            continue
    
    
    if all_original_imgs and all_attacked_imgs and all_recovered_imgs:
        original_batch = torch.cat(all_original_imgs, dim=0)
        attacked_batch = torch.cat(all_attacked_imgs, dim=0)
        recovered_batch = torch.cat(all_recovered_imgs, dim=0)
    else:
        print("No valid images were generated for some runs.")
        original_batch = None
        attacked_batch = None
        recovered_batch = None
    
    
    print("\n===== OVERALL RESULTS =====")
    
    
    print("\n--- Success Rates ---")
    attack_success_rate = (attack_success_count / NUM_RUNS) * 100
    defense_success_rate = (defense_success_count / NUM_RUNS) * 100
    print(f"Attack Success Rate: {attack_success_rate:.2f}% ({attack_success_count}/{NUM_RUNS})")
    print(f"Defense Success Rate: {defense_success_rate:.2f}% ({defense_success_count}/{NUM_RUNS})")
    
    
    print("\n--- LPIPS Evaluation (All Runs) ---")
    if attack_lpips_scores and defense_lpips_scores:
        avg_attack_lpips = np.mean(attack_lpips_scores)
        std_attack_lpips = np.std(attack_lpips_scores)
        avg_defense_lpips = np.mean(defense_lpips_scores)
        std_defense_lpips = np.std(defense_lpips_scores)
        
        print(f"Average Attack LPIPS (original vs attacked): {avg_attack_lpips:.4f} ± {std_attack_lpips:.4f}")
        print(f"Average Defense LPIPS (original vs recovered): {avg_defense_lpips:.4f} ± {std_defense_lpips:.4f}")
        print(f"Average Improvement: {avg_attack_lpips - avg_defense_lpips:.4f}")
    else:
        print("No LPIPS scores recorded.")
    
    
    print("\n--- Fréchet Inception Distance (FID) score ---")
    if original_batch is not None and attacked_batch is not None and recovered_batch is not None:
        try:
            
            fid_attack = calculate_fid_torcheval(original_batch.to(device), attacked_batch.to(device), device)
            print(f"FID (original vs attacked): {fid_attack:.4f}")
            
            fid_defense = calculate_fid_torcheval(original_batch.to(device), recovered_batch.to(device), device)
            print(f"FID (original vs recovered): {fid_defense:.4f}")
            
            print(f"FID Improvement: {fid_attack - fid_defense:.4f}")
        except Exception as e:
            print(f"Could not calculate FID: {e}")
    else:
        print("Skipping FID calculation: torcheval not available or no images generated.")
    
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(attack_lpips_scores, bins=20, alpha=0.7, color='red', label='Attack')
    plt.hist(defense_lpips_scores, bins=20, alpha=0.7, color='green', label='Defense')
    plt.axvline(x=LPIPS_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({LPIPS_THRESHOLD})')
    plt.title('LPIPS Distribution')
    plt.xlabel('LPIPS Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    runs = range(1, len(attack_lpips_scores) + 1)
    plt.plot(runs, attack_lpips_scores, 'r-', label='Attack LPIPS')
    plt.plot(runs, defense_lpips_scores, 'g-', label='Defense LPIPS')
    plt.axhline(y=LPIPS_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({LPIPS_THRESHOLD})')
    plt.title('LPIPS Scores per Run')
    plt.xlabel('Run')
    plt.ylabel('LPIPS Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lpips_metrics.png"))
    
    print(f"\nAll runs completed. Results saved to {output_dir}")