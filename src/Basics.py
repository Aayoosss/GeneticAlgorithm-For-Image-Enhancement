import streamlit as st


st.title("X-Ray Image Enhancer")

file = st.file_uploader("Please upload your image, and sit back....", type = ["jpg", "JPG", "PNG", "png", "JPEG", "jpeg"])
col1 = st.columns()
if file is not None:
    st.image(file, channels="RGB", output_format="auto", width= 400)

# import streamlit as st
# st.title("This is a title")
# st.title("_Streamlit_ is :blue[cool] :sunglasses:")