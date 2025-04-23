import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn.functional as F


def calculate_lpips(img1, img2):
    loss_fn = lpips.LPIPS(net='vgg')
    similarity = loss_fn(img1, img2)
    return similarity

def calculate_fid(real_images, generated_images, device):
    """Calculate Fréchet Inception Distance between two image sets"""
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    inception_model.fc = torch.nn.Identity()
    def get_features(images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        if images.min() < 0:
            images = (images + 1) / 2
            
        images = torch.clamp(images, 0, 1)
        
        with torch.no_grad():
            features = inception_model(images)
        return features.detach().cpu().numpy()

    real_features = get_features(real_images)
    gen_features = get_features(generated_images)

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def save_images(images, path=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    
    grid = torchvision.utils.make_grid(
        images.clamp(min=-1, max=1), 
        scale_each=True,             
        normalize=True             
    )
    
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    save_path = path if path else 'output.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()