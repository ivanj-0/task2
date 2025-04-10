import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn.functional as F


def calculate_lpips(img1, img2):
    loss_fn = lpips.LPIPS(net='vgg')
    similarity = loss_fn(img1, img2)
    return similarity

def calculate_fid(real_images, generated_images, device):
    """Calculate Fréchet Inception Distance between two image sets"""
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    inception_model.fc = torch.nn.Identity()  # Remove classification layer

    # Helper function to extract features
    def get_features(images):
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            
        # Resize to 299x299 as required by Inception v3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Scale images to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2
            
        # Ensure pixel values are in proper range
        images = torch.clamp(images, 0, 1)
        
        with torch.no_grad():
            features = inception_model(images)
        return features.detach().cpu().numpy()

    # Get features
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)

    # Calculate mean and covariance for both sets
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Check if complex numbers are generated (numerical issues)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def save_images(images, path=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()

    num_images, channels, height, width = images.shape

    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(num_images):
        img = images[i]
        plt.subplot(rows, cols, i + 1)
        if channels == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        elif channels == 3:
            plt.imshow(img.permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        plt.axis('off')
        plt.title(f'Image {i + 1}')

    plt.tight_layout()
    
    save_path = path if path else 'output.png'
    plt.savefig(save_path)