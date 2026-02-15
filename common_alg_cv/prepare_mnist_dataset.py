from pathlib import Path
import glob, os, shutil
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CFG

def _link_or_copy(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # hardlink (fast; same inode, no data copy)
        os.link(src, dst)
        os.remove(src)
    except OSError:
        # fallback (cross-device or FS doesn’t support hardlinks)
        shutil.copy2(src, dst)
        os.remove(src)

def prepare_mnist_dataset(base_path: Path = Path(__file__).parent, test_size: float = CFG.TEST_SIZE, random_state: int = CFG.SEED) -> None:
    images_dir = Path(base_path) / CFG.IMAGES_PATH
    print("Preparing MNIST dataset in:", images_dir)
    pattern = str(images_dir / "*.png")
    print(f"Using pattern: {pattern}")
    image_paths = sorted(glob.glob(pattern))
    print(f"Found {len(image_paths)} images.")

    train_paths, test_paths = train_test_split(
        image_paths, test_size=test_size, random_state=random_state
    )
    test_paths, val_paths = train_test_split(
        test_paths, test_size=0.5, random_state=random_state
    )

    train_dir = images_dir / "train"
    test_dir = images_dir / "test"
    val_dir = images_dir / "val"
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # parallelize I/O—safe for independent files
    workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for path in train_paths:
            dst = train_dir / Path(path).name
            futures.append(ex.submit(_link_or_copy, path, dst))
        for path in test_paths:
            dst = test_dir / Path(path).name
            futures.append(ex.submit(_link_or_copy, path, dst))
        for path in val_paths:
            dst = val_dir / Path(path).name
            futures.append(ex.submit(_link_or_copy, path, dst))
        for _ in as_completed(futures):
            pass

    print(f"Prepared MNIST dataset with {len(train_paths)} training and {len(test_paths)} testing images, and {len(val_paths)} validation images.")
