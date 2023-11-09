import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import cat

from models import Generator
from models import Discriminator
from models import weights_init

from utils import create_dataloader
from train import train_DCGAN

def get_arguments():
    parser = argparse.ArgumentParser(description="DCGAN settings.")
    
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizers")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs available")
    parser.add_argument("--folder_path", type=str, help="folder_path")
    parser.add_argument("--save_folder", type=str, default = "result", help="save folder path")
    return parser.parse_args()

def main(args):
    generater = Generator(args.ngpu, args.ngf, args.nz, args.nc)
    discriminator = Discriminator(args.ngpu, args.ngf, args.nz, args.nc)
    generater.apply(weights_init)
    discriminator.apply(weights_init)
    
    train_dataloader = create_dataloader(args.folder_path, args.image_size, args.batch_size, dim = args.nc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generater.parameters(), lr= args.lr , betas=(0.9 , 0.999))
    optimizerD = optim.Adam(generater.parameters(), lr= args.lr, betas=(0.9 , 0.999))
    train_DCGAN(train_dataloader, generater, discriminator, criterion, optimizerG, optimizerD, device,
                args.num_epochs, args.nz)
    print("\n\nDONE !")

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    print(">>> POONGSAN: Preparing for training...")
    main(args)