import torch
import matplotlib.pyplot as plt
import torchvision
import os

class DCGANGenerator():
    def __init__(self):
        self.use_gpu = True if torch.cuda.is_available() else False
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 
                                   'DCGAN', 
                                   pretrained=True, 
                                   useGPU=self.use_gpu)
    
    def generate_image(self, noise=None, num_images=64):
        if noise is None:
            noise, _ = self.model.buildNoiseData(num_images)
        
        with torch.no_grad():
            generated_images = self.model.test(noise)
        
        return generated_images
    
    def save_images(self, images, filepath="dcgan_generated.png"):
        grid = torchvision.utils.make_grid(images)
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Image saved to {os.path.abspath(filepath)}")

# Example usage
if __name__ == "__main__":
    generator = DCGANGenerator()
    num_images = 64
    noise, _ = generator.model.buildNoiseData(num_images)
    generated_images = generator.generate_image(noise)
    generator.save_images(generated_images, "dcgan_generated_images.png")