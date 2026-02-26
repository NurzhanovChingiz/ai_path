from typing import Any

from torch import Tensor, tensor
from torch.utils.data import Dataset

from ..read_img import read_image  # type: ignore[import-not-found]


class MNISTImageDataset(Dataset):
    def __init__(
            self,
            image_paths: list[str],
            labels: list[int],
            transform: Any = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor]:
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, tensor(label)
