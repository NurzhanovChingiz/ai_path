import cv2

def read_image(img_path):
    """
    Reads an image from the specified file path.

    Args:
        image_path (str): The path to the image file.
    Returns:
        image (numpy.ndarray): The image read from the file.
    """
    try:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error reading image at {img_path}: {e}")
        
if __name__ == "__main__":
    pass