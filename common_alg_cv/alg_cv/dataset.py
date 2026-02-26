"""Dataset classes for computer vision."""

from typing import Any

from torch import Tensor, tensor
from torch.utils.data import Dataset

from ..read_img import read_image  # type: ignore[import-not-found]


class MNISTImageDataset(Dataset):
    """PyTorch Dataset for MNIST images loaded from disk by file path."""
    def __init__(
            self,
            image_paths: list[str],
            labels: list[int],
            transform: Any = None) -> None:
        """Initialize dataset with image paths, labels, and optional transform."""
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor]:
        """Return image and label at the given index."""
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, tensor(label)
