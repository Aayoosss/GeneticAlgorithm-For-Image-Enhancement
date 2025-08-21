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
        logger.log("Image loaded successfully")
        return img
    except FileNotFoundError as e:
        logger.log("File Not Found Error: %s", e)
    except Exception as e:
        logger.log("Unexpected error occured while loading the image: %s", e)        


def main():
    try:
        image_path = "data/image75.jpeg"
        img = read_image(image_path)
        enhancer = Enhancer(img, cliplimit = 2.0, tilesize = (8,8))
        logger.log("Enhancement Initiated...........")
        enhanced_img = enhancer.RunGA(40, 100)
        logger.log("Enhancement Completed !")
        enhancer.compare_images(enhanced_img)
        
    except Exception as e:
        logger.log("Unexpected error occurred while enhancing the image: %s", e)

if __name__ == "__main__":
    main()


# enhancer.compare_images(enhanced_img)
# cv2.imshow("Enhanced Image", enhanced_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
