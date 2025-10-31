
import kagglehub
import os
import concurrent.futures as _cf
import asyncio
import shutil
from pathlib import Path
from config import CFG

from common_alg_ml.time_it import time_it

def download_mnist_data():
    # Delete existing data folder if exists
    shutil.rmtree(os.path.join(Path(__file__).parent, CFG.DOWNLOAD_PATH), ignore_errors=True)
    # Download latest version
    print('parents:', Path(__file__).parent)
    image_folder = os.path.join(Path(__file__).parent, CFG.DOWNLOAD_PATH)
    os.makedirs(image_folder, exist_ok=True)
    print("Downloading MNIST dataset to:", image_folder)
    path = kagglehub.dataset_download("animatronbot/mnist-digit-recognizer")
    shutil.move(path, image_folder)
    print("MNIST dataset downloaded to:", image_folder)
    image_folder = Path(__file__).parent / CFG.DOWNLOAD_PATH
    nested_folders = [p for p in image_folder.iterdir() if p.is_dir()]

    for folder in nested_folders:
        print(f"Flattening: {folder}")
        for item in folder.iterdir():
            target = image_folder / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(folder) # remove forlder after moving

    print("Data flattened into:", image_folder)

@time_it(iterations=1, label="Make Images from CSV")
def make_images_from_csv():
    import pandas as pd
    import cv2
    import numpy as np

    image_folder = Path(__file__).parent / CFG.DOWNLOAD_PATH
    csv_file = image_folder / "train.csv"
    df = pd.read_csv(csv_file)

    images_dir = image_folder / "images"
    os.makedirs(images_dir, exist_ok=True)
    data = df.to_numpy()
    async def write_one(data):
        index, row = data
        label = row[0]
        pixels = row[1:].astype(np.uint8)
        img = pixels.reshape(28, 28)
        img_path = images_dir / f"{index}_{label}.png"
        cv2.imwrite(str(img_path), img)
        
    max_workers = os.cpu_count()
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        for item in enumerate(data):
            tasks.append(loop.run_in_executor(executor, write_one, item))
        loop.run_until_complete(asyncio.gather(*tasks))
    print(f"Images saved to: {images_dir}")


if __name__ == "__main__":
    download_mnist_data()
    make_images_from_csv()