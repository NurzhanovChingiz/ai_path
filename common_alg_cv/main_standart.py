from config import CFG

from read_img import read_image
from pathlib import Path
import glob, os
import cv2
import random

from alg_cv.dataset import MNISTImageDataset
from alg_cv.train_cuda import train, CUDAPrefetcher
from alg_cv.test_fusion import test
import torch

from torchvision.transforms import v2
from torch.utils.data import DataLoader

import tqdm 
import time
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),

    ]) 
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    ])

if __name__ == "__main__":
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
    
    # Create datasets
    train_dataset = MNISTImageDataset(train_image_paths, train_labels, transform=train_transform, )
    test_dataset = MNISTImageDataset(test_image_paths, test_labels, transform=test_transform)
    val_dataset = MNISTImageDataset(val_image_paths, val_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    print(f"Number of training samples in dataset: {len(train_dataset)}")
    print(f"Number of testing samples in dataset: {len(test_dataset)}")
    print(f"Number of validation samples in dataset: {len(val_dataset)}")
    
    # look dataloader
    for images, labels in train_loader:
        print(f"Image batch shape: {images.size()}")
        print(f"Label batch shape: {labels.size()}")
        break
    
    if CFG.SHOW_IMG:
        n = 3 # 3 random images from dataset
        image_paths = random.sample(train_image_paths, n)
        for img_path in image_paths:
            img = read_image(str(img_path))
            cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # train
    prefetcher_train = CUDAPrefetcher(train_loader, CFG.DEVICE)
    prefetcher_test = CUDAPrefetcher(test_loader, CFG.DEVICE)
    prefetcher_val = CUDAPrefetcher(val_loader, CFG.DEVICE)
    for epoch in tqdm.tqdm(range(CFG.EPOCHS), desc="Epochs"):
        start = time.time()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        prefetcher_train.reset()
        print(f"Epoch [{epoch+1}/{CFG.EPOCHS}]")
        train(CFG.MODEL, train_loader, CFG.LOSS_FN, CFG.OPTIMIZER, CFG.DEVICE, prefetcher=prefetcher_train)
        test(CFG.MODEL, val_loader, CFG.LOSS_FN, CFG.DEVICE, mode="Validation")
        end = time.time()
        print(f"Epoch [{epoch+1}/{CFG.EPOCHS}] completed in {end - start:.2f} seconds")
        
    test(CFG.MODEL, test_loader, CFG.LOSS_FN, CFG.DEVICE, mode="Test")