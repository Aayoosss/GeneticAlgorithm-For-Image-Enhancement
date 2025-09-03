from enhancer import Enhancer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

log_dir_path = "logs"
os.makedirs(log_dir_path, exist_ok = True)


logger = logging.Logger("app.py")
logger.setLevel("DEBUG")

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel("DEBUG")

file_path = os.path.join(log_dir_path,"app.log")
fileHandler = logging.FileHandler(file_path)
fileHandler.setLevel("DEBUG")

Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileHandler.setFormatter(Formatter)
consoleHandler.setFormatter(Formatter)

logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def read_image(image_path: str) -> np.array:
    try:
        img = cv2.imread(image_path)
        logger.info("Image loaded successfully")
        return img
    except Exception as e:
        logger.error("Unexpected error occured while loading the image: %s", e)        


def main():
    try:
        image_path = r"data/image75.jpeg"
        img = read_image(image_path)
        enhancer = Enhancer(img, cliplimit = 2.0)
        logger.info("Enhancement Initiated...........")
        enhanced_img = enhancer.RunGA(20, 50)
        logger.info("Enhancement Completed !")
        gentable = enhancer.get_gentable()
        grid_size = enhancer.get_best_chromosome()
        # print(gentable)
        enhancer.compare_images(enhanced_img)
        
    except Exception as e:
        logger.error("Unexpected error occurred while enhancing the image: %s", e)
        raise

if __name__ == "__main__":
    main()
