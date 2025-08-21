import streamlit as st
import cv2
import numpy as np
from enhancer import Enhancer

st.title("X-Ray Image Enhancer")

file = st.file_uploader("Please upload your image, and sit back....", type = ["jpg", "JPG", "PNG", "png", "JPEG", "jpeg","AVIF","avif"])
with st.form("My_form"):
    population_size = st.selectbox("Select Population Size: ", [20,40,60,100], index = None)
    generations = st.selectbox("Select number of generations: ",  [40,50,100,150,200,250,500,1000], index = None)
    cliplimit = st.slider("Set cliplimit: ", 1, 100)
    submitted = st.form_submit_button("Submit")
    
if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    file = cv2.imdecode(file_bytes, 0)
    st.image(file, channels="RGB", output_format="auto", width= 200, caption= "Uploade image")
    if submitted:
            image = Enhancer(file, cliplimit = cliplimit)
            enhanced = image.RunGA(population_size=population_size, generations = generations) 
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Original Image")
                st.image(file)
                
            with col2:
                st.header("Enhanced Image")
                st.image(enhanced)

# import streamlit as st
# st.title("This is a title")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")