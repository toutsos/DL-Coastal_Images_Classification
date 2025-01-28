import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load training data without normalization
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor()
])
train_dataset = datasets.ImageFolder(root="/home/angelos.toutsios.gr/data/CS4321/HW1/teamsmt/data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=False)

# Calculate mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in train_loader:
    for i in range(3):
        mean[i] += images[:, i, :, :].mean()
        std[i] += images[:, i, :, :].std()

mean /= len(train_loader)
std /= len(train_loader)

print(f"Calculated mean: {mean.numpy()}, std: {std.numpy()}")

# 299x299 Images: Calculated mean: [0.48039493 0.47629836 0.4453485], std: [0.1743606  0.15852651 0.16594319]

# 224x224 Images: Calculated mean: [0.48039714 0.476299   0.44534707], std: [0.16886398 0.15295851 0.16063267}