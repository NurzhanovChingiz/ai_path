from sklearn.model_selection import train_test_split
from config import CFG
import os
import glob
from pathlib import Path
import shutil

def prepare_mnist_dataset(base_path= Path(__file__).parent, test_size=CFG.TEST_SIZE, random_state=CFG.SEED):
    
    
    images_dir = Path(base_path) / CFG.IMAGES_PATH
    print("Preparing MNIST dataset in:", images_dir)
    pattern = str(images_dir / "*.png")
    print(f"Using pattern: {pattern}")
    image_paths = sorted(glob.glob(pattern))
    print(f"Found {len(image_paths)} images.")
    train_paths, val_paths = train_test_split(
        image_paths, test_size=test_size, random_state=random_state
    )

    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for path in train_paths:
        shutil.move(path, train_dir / Path(path).name)

    for path in val_paths:
        shutil.move(path, val_dir / Path(path).name)

    print(f"Prepared MNIST dataset with {len(train_paths)} training and {len(val_paths)} validation images.")