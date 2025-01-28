from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a basic transformation to convert to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset
dataset = datasets.ImageFolder(root="/home/angelos.toutsios.gr/data/CS4321/HW1/teamsmt/data/train", transform=transform)

# Check image sizes
for img, label in dataset:
    print(f"Image size: {img.shape}")  # Shape format: (Channels, Height, Width)
    break  # Remove break to print all images