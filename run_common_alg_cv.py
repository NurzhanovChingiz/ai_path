from common_alg_cv.mnist_download import KaggleDatasetDownloader, make_images_from_csv # type: ignore[import-not-found]
from common_alg_cv.prepare_mnist_dataset import prepare_mnist_dataset # type: ignore[import-not-found]
from config import CFG
import time


def run_data_preparation() -> None:
    mnist_downloader = KaggleDatasetDownloader(
        dataset_id=CFG.DATASET, path_to_save=CFG.IMAGES_PATH)
    mnist_downloader.run()
    make_images_from_csv()
    prepare_mnist_dataset()


if __name__ == "__main__":
    start_time = time.time()
    if CFG.DATA_PREPARATION:
        run_data_preparation()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
