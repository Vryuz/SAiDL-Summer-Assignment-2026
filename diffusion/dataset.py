import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub

class LandscapeDataset(Dataset):
    """
    Dataset class for loading the Arnaud58 landscape pictures dataset.
    Automatically handles corrupt images by returning a zero tensor.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # If an image is corrupt, return a blank tensor to prevent crashing
            return torch.zeros((3, 256, 256))

def get_dataloader(batch_size=8, num_workers=0):
    """
    Downloads the Kaggle landscape dataset and returns a PyTorch DataLoader.
    Images are resized to 256x256 and normalized to [-1, 1] for diffusion.
    """
    print("Downloading/Locating dataset via kagglehub...")
    path = kagglehub.dataset_download("arnaud58/landscape-pictures")
    print(f"Dataset located at: {path}")

    # Gather all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
    ])

    dataset = LandscapeDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dataloader

if __name__ == "__main__":
    loader = get_dataloader(batch_size=4, num_workers=0)
    batch = next(iter(loader))
    print(f"Sample batch shape: {batch.shape}")
    print(f"Sample batch min: {batch.min()}, max: {batch.max()}")
