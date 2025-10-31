from sklearn.model_selection import train_test_split
import mnist_download

def prepare_mnist_dataset(base_path, test_size=0.2, random_state=42):
    mnist_download.download_mnist_data()
    mnist_download.make_images_from_csv()