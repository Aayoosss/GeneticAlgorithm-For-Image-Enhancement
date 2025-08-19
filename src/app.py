import streamlit as st
from enhancer import Enhancer
import cv2
import matplotlib.pyplot as plt

image_path = "data/image75.jpeg"
img = cv2.imread(image_path)

# Example enhancement (convert to grayscale + CLAHE)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
enhancer = Enhancer(img, cliplimit = 2.0, tilesize = (8,8))
enhanced_img = enhancer.RunGA(40, 100)
enhancer.compare_images(enhanced_img)



# enhancer.compare_images(enhanced_img)
# cv2.imshow("Enhanced Image", enhanced_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
