import numpy as np
import torch
import lpips
import os
from pgan import PGANGenerator
from dcgan import DCGANGenerator
from utils import save_images
from tqdm import tqdm
import matplotlib.pyplot as plt


def black_box_attack(generator, perceptual_loss_model, num_iterations, step_size, device, num_perturbations=10):
    z, _ = generator.model.buildNoiseData(1)
    z = z.to(device)
    z.requires_grad_(False)

    original_image = generator.generate_image(noise=z).detach().to(device)

    history = []
    pbar = tqdm(range(num_iterations), desc=f"Attack iters={num_iterations}, step={step_size}, perturbs={num_perturbations}")
    for i in pbar:
        perturbations, _ = generator.model.buildNoiseData(num_perturbations)
        perturbations = perturbations.to(device)

        batch_z = z.repeat(num_perturbations, 1) + step_size * perturbations
        perturbed_images = generator.generate_image(noise=batch_z).detach().to(device)

        original_batch = original_image.repeat(num_perturbations, 1, 1, 1)

        with torch.no_grad():
            dists = perceptual_loss_model(original_batch, perturbed_images).squeeze()
        scores = dists.cpu().numpy()
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        history.append(best_score)

        z = z + step_size * perturbations[best_idx:best_idx+1]
        pbar.set_postfix({"LPIPS": f"{best_score:.4f}"})

    attacked_image = generator.generate_image(noise=z).detach().to(device)
    return z, original_image, attacked_image, history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss.eval()

    generator = DCGANGenerator()
    try:
        generator.model.to(device)
    except Exception:
        pass
    print("Generator loaded and moved to device.")

    configs = [
        {'iterations': 50,  'step': 0.005, 'perts': 10},
        {'iterations': 100, 'step': 0.01,  'perts': 20},
        {'iterations': 200, 'step': 0.02,  'perts': 20},
    ]

    all_histories = {}
    final_scores = {}

    for cfg in configs:
        key = f"iters{cfg['iterations']}_step{cfg['step']}_perts{cfg['perts']}"
        z_fin, orig, att, hist = black_box_attack(
            generator, perceptual_loss,
            num_iterations=cfg['iterations'],
            step_size=cfg['step'],
            device=device,
            num_perturbations=cfg['perts']
        )
        all_histories[key] = hist

        with torch.no_grad():
            dist = perceptual_loss(orig, att).item()
        final_scores[key] = dist

        comp = torch.cat([orig.cpu(), att.cpu()], dim=0)
        fname = f"attack_{key}.png"
        save_images(comp, path=fname)
        print(f"Saved comparison: {fname}")

    plt.figure()
    for key, hist in all_histories.items():
        plt.plot(hist, label=key)
    plt.xlabel('Iteration')
    plt.ylabel('LPIPS Distance')
    plt.title('Attack Progression')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lpips_progression.png')
    print("Saved plot: lpips_progression.png")