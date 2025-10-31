from common_alg_cv.mnist_download import KaggleDatasetDownloader, make_images_from_csv
from common_alg_cv.prepare_mnist_dataset import prepare_mnist_dataset
from config import CFG
import time 
if __name__ == "__main__":
    start_time = time.time()
    mnist_downloader = KaggleDatasetDownloader(dataset_id=CFG.DATASET)
    mnist_downloader.run()
    make_images_from_csv()
    prepare_mnist_dataset()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")