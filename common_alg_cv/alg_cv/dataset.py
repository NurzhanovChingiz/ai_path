from torch.utils.data import Dataset
from read_img import read_image

class MNISTImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label