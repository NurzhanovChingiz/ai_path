from read_img import read_image
from pathlib import Path
import glob, os
import cv2
import random
from config import CFG

from alg_cv.clear_gpu import clear_memory

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    ])

class CachedDataset(Dataset):
    def __init__(self, base_dataset):
        self.data = [base_dataset[i] for i in range(len(base_dataset))] 

        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
        
if __name__ == "__main__":
    clear_memory()
    print(__file__)
    main_folder = os.getcwd()
    print("main_path:", main_folder)
    train_images_dir = Path(__file__).parent / "mnist_data/images/train"
    test_images_dir = Path(__file__).parent / "mnist_data/images/test"
    val_images_dir = Path(__file__).parent / "mnist_data/images/val"
    print("read_img.py path:", Path(__file__).parent)
    train_pattern = str(train_images_dir / "*.png")
    test_pattern = str(test_images_dir / "*.png")
    val_pattern = str(val_images_dir / "*.png")
    train_image_paths = sorted(glob.glob(train_pattern))
    test_image_paths = sorted(glob.glob(test_pattern))
    val_image_paths = sorted(glob.glob(val_pattern))
    print(f"Number of training images: {len(train_image_paths)}")
    print(f"Number of testing images: {len(test_image_paths)}")
    print(f"Number of validation images: {len(val_image_paths)}")
    train_labels = [int(Path(p).name.split("_")[1].split(".")[0]) for p in train_image_paths]
    test_labels = [int(Path(p).name.split("_")[1].split(".")[0]) for p in test_image_paths]
    val_labels = [int(Path(p).name.split("_")[1].split(".")[0]) for p in val_image_paths]
    
    if CFG.SHOW_IMG:
        n = 3 # 3 random images from dataset
        image_paths = random.sample(train_image_paths, n)
        for img_path in image_paths:
            img = read_image(str(img_path))
            cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()