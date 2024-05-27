import os
import urllib.request
import zipfile
import tarfile
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url

# Define dataset directories
data_dir = "data"
cifar10_dir = os.path.join(data_dir, "cifar10")
gtsrb_dir = os.path.join(data_dir, "gtsrb")
self_collected_dir = os.path.join(data_dir, "self_collected")

# Ensure dataset directories exist
os.makedirs(cifar10_dir, exist_ok=True)
os.makedirs(gtsrb_dir, exist_ok=True)
os.makedirs(self_collected_dir, exist_ok=True)

# Function to download and extract CIFAR-10 dataset
def download_cifar10(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# Function to download and extract GTSRB dataset
def download_gtsrb(data_dir):
    url = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    gtsrb_zip_path = os.path.join(data_dir, "GTSRB_Final_Training_Images.zip")
    download_url(url, data_dir, "GTSRB_Final_Training_Images.zip")
    with zipfile.ZipFile(gtsrb_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(gtsrb_zip_path)

# Function to download and extract self-collected object detection dataset
def download_self_collected(data_dir):
    url = "http://example.com/self_collected_dataset.zip"  # Replace with actual URL
    self_collected_zip_path = os.path.join(data_dir, "self_collected_dataset.zip")
    download_url(url, data_dir, "self_collected_dataset.zip")
    with zipfile.ZipFile(self_collected_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(self_collected_zip_path)

# Download and prepare datasets
download_cifar10(cifar10_dir)
download_gtsrb(gtsrb_dir)
download_self_collected(self_collected_dir)

print("Datasets downloaded and extracted successfully.")
