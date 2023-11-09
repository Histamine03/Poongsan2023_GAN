from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, dim=3, transform=None):
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if self.dim == 3:
            image = Image.open(image_path).convert('RGB')  # RGB 형식으로 변환
        elif self.dim == 1:
            image = Image.open(image_path).convert('L')  # 흑백 형식으로 변환
        else:
            raise ValueError("dim should be either 1 (grayscale) or 3 (RGB)")
    
        if self.transform:
            image = self.transform(image)
    
        return image

def create_dataloader(folder_path, image_size, batch_size, dim=3):
    if dim == 3:
        preprocess = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dim == 1:
        preprocess = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        raise ValueError("dim should be either 1 (grayscale) or 3 (RGB)")
    
    dataset = CustomImageDataset(folder_path, dim=dim, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader