import os
import urllib.request
import zipfile
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url

# Define dataset directories
data_dir = "data"
cifar10_dir = os.path.join(data_dir, "cifar10")
gtsrb_dir = os.path.join(data_dir, "gtsrb")

# Ensure dataset directories exist
os.makedirs(cifar10_dir, exist_ok=True)
os.makedirs(gtsrb_dir, exist_ok=True)


# Function to download and extract CIFAR-10 dataset
def download_cifar10(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )


# Function to download and extract GTSRB dataset
def download_gtsrb(data_dir):
    url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea82a1e84fad7a77/GTSRB_Final_Training_Images.zip"
    gtsrb_zip_path = os.path.join(data_dir, "GTSRB_Final_Training_Images.zip")
    download_url(url, data_dir, "GTSRB_Final_Training_Images.zip")
    with zipfile.ZipFile(gtsrb_zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, "GTSRB"))
    os.remove(gtsrb_zip_path)


# Download and prepare datasets
download_cifar10(cifar10_dir)
download_gtsrb(gtsrb_dir)

print("Datasets downloaded and extracted successfully.")
