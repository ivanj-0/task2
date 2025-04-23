import numpy as np
import torch
import lpips
import cma
import os
from pgan import PGANGenerator
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm
import matplotlib.pyplot as plt



def cma_es_attack(generator, perceptual_loss_model, num_iterations, sigma, device):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    generator = DCGANGenerator()
    print("Generator loaded.")

    iteration_list = [200]
    sigma_list = [0.005, 0.01, 0.02]

    results = []

    for num_iterations in iteration_list:
        for sigma in sigma_list:
            print(f"\nRunning attack: Iterations={num_iterations}, Sigma={sigma}")

            score_history = []

            def track_score(generator, perceptual_loss_model, num_iterations, sigma, device):
                z, _ = generator.model.buildNoiseData(1)
                z = z.to(device)
                z.requires_grad_(False)

                original_image = generator.generate_image(noise=z).detach()
                z_np = z.cpu().numpy().reshape(-1)
                es = cma.CMAEvolutionStrategy(z_np, sigma)

                best_score = -np.inf
                best_z = z.clone()

                scores_per_iteration = []

                for _ in tqdm(range(num_iterations), desc=f"CMA-ES [{sigma}, {num_iterations}]"):
                    solutions_np = es.ask()
                    solutions_list = [torch.tensor(sol, dtype=torch.float32, device=device).unsqueeze(0) for sol in solutions_np]
                    perturbed_z_batch = torch.cat(solutions_list, dim=0)
                    perturbed_images_batch = generator.generate_image(noise=perturbed_z_batch).detach()
                    original_image_batch = original_image.repeat(len(solutions_list), 1, 1, 1)

                    with torch.no_grad():
                        dists = perceptual_loss_model(original_image_batch.to(device), perturbed_images_batch.to(device)).squeeze()

                    scores = dists.cpu().numpy()
                    es.tell(solutions_np, -scores)

                    current_best_score = np.max(scores)
                    scores_per_iteration.append(current_best_score)

                    if current_best_score > best_score:
                        best_score = current_best_score
                        best_z = solutions_list[np.argmax(scores)].detach().clone()

                final_z = torch.tensor(best_z, dtype=torch.float32, device=device)
                attacked_image = generator.generate_image(noise=final_z).detach()

                return final_z, original_image, attacked_image, scores_per_iteration, best_score

            z_final, original_image, attacked_image, score_list, best_lpips = track_score(
                generator=generator,
                perceptual_loss_model=perceptual_loss,
                num_iterations=num_iterations,
                sigma=sigma,
                device=device
            )

            results.append({
                "iterations": num_iterations,
                "sigma": sigma,
                "score_list": score_list,
                "original_image": original_image,
                "attacked_image": attacked_image,
                "final_lpips": best_lpips
            })

            comparison_tensor = torch.cat([original_image.cpu(), attacked_image.cpu()], dim=0)
            out_name = f"attack_comparison_iter{num_iterations}_sigma{sigma}.png"
            save_images(comparison_tensor, path=out_name)

    for i, res in enumerate(results):
        res['sigma'] = k[i]
    for res in results:
        plt.plot(res["score_list"], label=f"Iter={res['iterations']}, Sigma={res['sigma']}")
    plt.title("LPIPS Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("LPIPS Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("lpips_score_over_iterations.png")
    plt.close()
    print("Plots saved.") 
