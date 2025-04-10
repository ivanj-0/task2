import torch
import matplotlib.pyplot as plt
import torchvision
import os

class PGANGenerator():
    def __init__(self):
        self.use_gpu = True if torch.cuda.is_available() else False
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                           'PGAN', model_name='celebAHQ-512',
                           pretrained=True, useGPU=self.use_gpu)
    
    def generate_image(self, noise=None, num_images=4):
        if noise is None:
            noise, _ = self.model.buildNoiseData(num_images)
        
        with torch.no_grad():
            generated_images = self.model.test(noise)
        
        return generated_images
    
    def save_images(self, images, filepath="generated_image.png"):
        grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Image saved to {os.path.abspath(filepath)}")

# Example usage
if __name__ == "__main__":
    generator = PGANGenerator()
    num_images = 4
    noise, _ = generator.model.buildNoiseData(num_images)
    generated_images = generator.generate_image(noise)
    generator.save_images(generated_images, "pgan_generated_images.png")