from read_img import read_image
from pathlib import Path
import glob, os
import cv2
import random
if __name__ == "__main__":
    print(__file__)
    main_folder = os.getcwd()
    print("main_path:", main_folder)
    images_dir = Path(__file__).parent / "mnist_data/images"
    print("read_img.py path:", Path(__file__).parent)
    pattern = str(images_dir / "*.png")
    image_paths = sorted(glob.glob(pattern))

    n = 3 # 3 random images from dataset
    image_paths = random.sample(image_paths, n)
    for img_path in image_paths:
        img = read_image(str(img_path))
        cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()