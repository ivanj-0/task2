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
    device=None
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

        if iteration % 10 == 0:
            print(f"Generation {iteration}, best fitness: {best_gen_fitness:.6f}, best LPIPS: {best_gen_lpips:.6f}")

        iteration += 1

    best_solution = es.result[0]
    best_fitness = es.result[1]

    print(f"Final best fitness: {best_fitness:.6f}")

    best_z = torch.tensor(best_solution.reshape(z_init.shape),
                          device=device,
                          dtype=torch.float32)

    with torch.no_grad():
        best_img = generator.generate_image(best_z)

    print(f"CMA-ES optimization complete.")

    return best_img, best_z, lpips_history


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        generator = DCGANGenerator()
    except NameError:
        print("Error: PGANGenerator not defined. Please ensure it's imported correctly.")
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

    LATENT_DEFENSE_OUTPUT_FILENAME = "defense_comparison_latent_defense_cma_es.png"
    LPIPS_PLOT_FILENAME = "defense_lpips_history.png"

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    print("Performing CMA-ES attack...")
    try:
        z_adv, original_img, attacked_img = cma_es_attack(
            generator=generator,
            perceptual_loss_model=perceptual_loss,
            num_iterations=ATTACK_ITERATIONS,
            sigma=ATTACK_SIGMA,
            device=device
        )
    except NameError:
        print("Error: cma_es_attack not defined. Please ensure it's imported correctly.")
        exit()
    except Exception as e:
        print(f"Error during attack: {e}")
        exit()

    print("Attack finished.")
    adversarial_latent_shape = z_adv.shape

    recovered_img_latent, z_recovered_latent, defense_lpips_history = improved_latent_defense_cma(
        generator=generator,
        original_img=original_img,
        z_init=z_adv,
        latent_dim=latent_dim,
        population_size=LATENT_DEFENSE_POP_SIZE,
        max_iterations=LATENT_DEFENSE_STEPS,
        sigma0=LATENT_DEFENSE_SIGMA0,
        weight_lpips=LATENT_DEFENSE_WEIGHT_LPIPS,
        weight_l2=LATENT_DEFENSE_WEIGHT_L2,
        device=device
    )
    print("Latent defense applied.")

    with torch.no_grad():
        original_img = original_img.to(device)
        attacked_img = attacked_img.to(device)
        recovered_img_latent = recovered_img_latent.to(device)

        lpips_attacked = perceptual_loss(original_img, attacked_img).mean().item()
        lpips_latent = perceptual_loss(original_img, recovered_img_latent).mean().item()

        l2_attacked = F.mse_loss(original_img, attacked_img).item()
        l2_latent = F.mse_loss(original_img, recovered_img_latent).item()

    print("\nDefense Comparison Metrics:")
    print(f"LPIPS - Attacked: {lpips_attacked:.4f}, Latent Defense: {lpips_latent:.4f}")
    print(f"L2    - Attacked: {l2_attacked:.4f}, Latent Defense: {l2_latent:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(defense_lpips_history)), defense_lpips_history)
    plt.title("LPIPS Score over Defense Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("LPIPS Score")
    plt.grid(True)
    plt.savefig(LPIPS_PLOT_FILENAME)
    print(f"Saved LPIPS history plot to {LPIPS_PLOT_FILENAME}")

    original_img_cpu = original_img.cpu()
    attacked_img_cpu = attacked_img.cpu()
    recovered_img_latent_cpu = recovered_img_latent.cpu()

    latent_comparison = torch.cat([original_img_cpu, attacked_img_cpu, recovered_img_latent_cpu], dim=0)
    print(f"Saving Latent defense comparison image to {LATENT_DEFENSE_OUTPUT_FILENAME}...")
    try:
        save_images(latent_comparison, path=LATENT_DEFENSE_OUTPUT_FILENAME)
    except NameError:
        print("Error: save_images not defined. Cannot save comparison image.")
    except Exception as e:
        print(f"Error saving comparison image: {e}")


    print("Done.")