import kagglehub
import os
import concurrent.futures as _cf
import shutil
from pathlib import Path
from config import CFG
import pandas as pd
import numpy as np
import cv2

class KaggleDatasetDownloader:
    def __init__(self, dataset_id: str) -> None:
        '''
        Initializes the KaggleDatasetDownloader with the specified dataset ID.
        Args:
            dataset_id (str): The Kaggle dataset identifier (e.g., "username/dataset-name").
        '''
        self.dataset_id = dataset_id
        assert self.dataset_id, "Dataset ID must be provided."
        assert isinstance(self.dataset_id, str), "Dataset ID must be a string."
        assert CFG and CFG.DOWNLOAD_PATH, "CFG and CFG.DOWNLOAD_PATH must be set in config."
        
    def resolve_base_dir(self) -> Path:
        '''
        Resolves the base directory for the dataset download.
        Returns:
            Path: The base directory path.
        '''
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd()
        return base_dir

    def clean_target_folder(self, data_folder: str) -> None:
        '''
        Cleans the target folder by removing the existing dataset files.
        Args:
            data_folder (str): The path to the dataset folder.
        '''
        shutil.rmtree(os.path.join(self.resolve_base_dir(), CFG.DOWNLOAD_PATH), ignore_errors=True)
        os.makedirs(data_folder, exist_ok=True)
    
    def download_dataset(self, data_folder: str) -> Path:
        '''
        Downloads the dataset from Kaggle and returns the source root path.
        Args:
            data_folder (str): The path to the folder.
        '''
        print(f"Downloading {self.dataset_id} dataset to:", data_folder)
        src_root = Path(kagglehub.dataset_download(self.dataset_id))
        if not src_root.exists():
            raise FileNotFoundError(f"Downloaded path not found: {src_root}")
        return src_root

    def _overwrite_move(self, src: Path, dst: Path) -> None:
        '''
        Moves a file or directory from src to dst, overwriting dst if it exists.
        Args:
            src (Path): Source path.
            dst (Path): Destination path.
        '''
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        try:
            os.replace(str(src), str(dst))
        except OSError:
            shutil.move(str(src), str(dst))

    def _move_items(self, src_root: Path, data_folder_path: Path) -> None:
        '''
        Moves all items from src_root to data_folder_path using multithreading.
        Args:
            src_root (Path): Source root path.
            data_folder_path (Path): Destination folder path.
        '''
        items = list(src_root.iterdir()) if src_root.is_dir() else [src_root]
        def _move_one(p: Path):
            target = data_folder_path / p.name
            self._overwrite_move(p, target)
        with _cf.ThreadPoolExecutor() as executor:
            list(executor.map(_move_one, items))
    
    def cleanup_src_root(self, src_root: Path) -> None:
        '''
        Cleans up the source root directory after moving items.
        Args:
            src_root (Path): Source root path.
        '''
        try:
            if src_root.is_dir():
                for leftover in src_root.iterdir():
                    if leftover.is_dir():
                        shutil.rmtree(leftover, ignore_errors=True)
                    else:
                        try:
                            leftover.unlink()
                        except FileNotFoundError:
                            pass
                try:
                    src_root.rmdir()
                except OSError:
                    pass
        except Exception:
            pass
    
    def check_files_exist(self, data_folder: str) -> None:
        '''
        Checks if the dataset files already exist in the target folder.
        Args:
            data_folder (str): The path to the dataset folder.
        
        '''
        contents = list(Path(data_folder).glob("*"))
        if not contents:
            raise RuntimeError(f"No files found after move in {data_folder}")
    
    def run(self) -> None:
        '''
        Executes the dataset download and organization process.
        '''
        base_dir = self.resolve_base_dir()
        data_folder = os.path.join(base_dir, CFG.DOWNLOAD_PATH)
        self.clean_target_folder(data_folder)
        src_root = self.download_dataset(data_folder)
        data_folder_path = Path(data_folder)
        self._move_items(src_root, data_folder_path)
        self.cleanup_src_root(src_root)
        self.check_files_exist(data_folder)
        print(f"{self.dataset_id} dataset downloaded and organized.")

def make_images_from_csv():
    image_folder = Path(__file__).parent / CFG.DOWNLOAD_PATH
    csv_file = image_folder / "train.csv"
    df = pd.read_csv(csv_file, dtype=np.uint8, engine="c")
    data = df.to_numpy(copy=False)  # shape: (N, 1+784), dtype=uint8
    images_dir = image_folder / "images"
    
    os.makedirs(images_dir, exist_ok=True)
    labels = data[:, 0]                       # (N,)
    imgs = data[:, 1:].reshape(-1, 28, 28)    # (N, 28, 28), uint8
    paths = [str(images_dir / f"{i}_{int(labels[i])}.png") for i in range(len(labels))]
    imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    def write_one(i: int):
        p = paths[i]
        cv2.imwrite(p, imgs[i], imwrite_params)
        
    max_workers = os.cpu_count() or 4

    # Larger chunksize cuts executor overhead for big N
    with _cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(write_one, range(len(labels)), chunksize=512))

    print(f"Images saved to: {images_dir}")
