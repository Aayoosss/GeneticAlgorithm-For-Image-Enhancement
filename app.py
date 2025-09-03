import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from src.enhancer import Enhancer

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "avif"]
MAX_IMAGE_DISPLAY_WIDTH = 400

def load_and_validate_image(uploaded_file) -> Optional[np.ndarray]:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            st.error("Failed to decode image. Please upload a valid image file.")
            return None
            
        if image.shape[0] < 50 or image.shape[1] < 50:
            st.error("Image is too small. Please upload an image with minimum 50x50 pixels.")
            return None
            
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def validate_parameters(population_size: int, generations: int, cliplimit: float) -> bool:
    if population_size is None or generations is None:
        st.error("Please select both population size and number of generations.")
        return False
    
    if cliplimit <= 0 or cliplimit > 100:
        st.error("Clip limit must be between 1 and 100.")
        return False
        
    return True

def enhance_image_with_progress(image: np.ndarray, population_size: int, 
                              generations: int, cliplimit: float) -> Optional[np.ndarray]:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing image enhancer...")
        progress_bar.progress(10)
        
        enhancer = Enhancer(image, cliplimit=cliplimit)
        
        status_text.text("Running genetic algorithm to find optimal parameters...")
        progress_bar.progress(30)
        
        enhanced_image = enhancer.RunGA(population_size=population_size, generations=generations)
        
        if enhanced_image is None:
            st.error("Enhancement failed. Please try with different parameters.")
            return None
            
        progress_bar.progress(90)
        status_text.text("Finalizing enhancement...")
        time.sleep(0.5)
        
        progress_bar.progress(100)
        status_text.text("Enhancement completed successfully!")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        return enhanced_image, enhancer.get_best_chromosome()
        
    except Exception as e:
        st.error(f"Enhancement failed: {str(e)}")
        return None

def display_results(original_image: np.ndarray, enhanced_image: np.ndarray, 
                   population_size: int, generations: int, cliplimit: float, grid_length: int, grid_width: int):
    st.success("✅ Image enhanced successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, width=MAX_IMAGE_DISPLAY_WIDTH, caption="Original X-ray image")
        
    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced_image, width=MAX_IMAGE_DISPLAY_WIDTH, caption="Enhanced X-ray image")
    
    st.subheader("Enhancement Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Population Size", population_size)
    with col2:
        st.metric("Generations", generations)
    with col3:
        st.metric("Clip Limit", cliplimit)
    with col4:
        st.metric("Grid length",  grid_length)
        st.metric("Grid width", grid_width)
    
    st.subheader("Download Enhanced Image")
    if st.button("💾 Prepare Download"):
        success, encoded_image = cv2.imencode('.png', enhanced_image)
        if success:
            st.download_button(
                label="Download Enhanced Image",
                data=encoded_image.tobytes(),
                file_name="enhanced_xray.png",
                mime="image/png"
            )

def main():
    st.set_page_config(
        page_title="X-Ray Image Enhancer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 X-Ray Image Enhancer")
    st.markdown("---")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This tool uses genetic algorithms to automatically find the optimal "
            "parameters for enhancing X-ray images using CLAHE (Contrast Limited "
            "Adaptive Histogram Equalization)."
        )
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload an X-ray image
        2. Configure enhancement parameters
        3. Click Submit to process
        4. Compare results and download
        """)
    
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image to enhance",
        type=SUPPORTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG, PNG, AVIF"
    )
    
    if uploaded_file is not None:
        image = load_and_validate_image(uploaded_file)
        
        if image is not None:
            st.success(f"✅ Image loaded successfully! Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
            
            with st.expander("🔍 Preview Uploaded Image", expanded=True):
                st.image(image, width=300, caption="Uploaded X-ray image")
            
            st.header("⚙️ Enhancement Parameters")
            st.warning("⚠️ **Processing Time Warning**: Larger population sizes and more generations will significantly increase processing time. Consider starting with smaller values for testing.")

            with st.form("enhancement_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    population_size = st.selectbox(
                        "Population Size",
                        options=[20, 40, 60, 100],
                        index=1,
                        help="Larger populations may find better solutions but take longer"
                    )
                    
                    cliplimit = st.slider(
                        "Clip Limit",
                        min_value=1.0,
                        max_value=10.0,
                        value=2.0,
                        step=0.1,
                        help="Controls contrast enhancement strength"
                    )
                
                with col2:
                    generations = st.selectbox(
                        "Number of Generations",
                        options=[40, 50, 100, 150, 200, 250, 500, 1000],
                        index=2,
                        help="More generations may find better solutions but take longer"
                    )
                
                estimated_time = (population_size * generations) / 10000
                st.info(f"⏱️ Estimated processing time: ~{estimated_time:.1f} seconds")
                
                submitted = st.form_submit_button(
                    "🚀 Enhance Image",
                    type="primary",
                    use_container_width=True
                )
            
            if submitted:
                if validate_parameters(population_size, generations, cliplimit):
                    with st.spinner("Processing image enhancement..."):
                        enhanced_image, grid_size = enhance_image_with_progress(
                            image, population_size, generations, cliplimit
                        )
                    
                    if enhanced_image is not None:
                        display_results(
                            image, enhanced_image,
                            population_size, generations, cliplimit, grid_size[0], grid_size[1]
                        )
    else:
        st.info("👆 Please upload an X-ray image to get started")
        
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - **Population Size**: Start with 40-60 for balanced performance
            - **Generations**: Use 100-200 for good results, 500+ for optimal quality
            - **Clip Limit**: Lower values (1-3) for subtle enhancement, higher for stronger contrast
            - **Image Quality**: Higher resolution images generally produce better results
            - **Processing Time**: Larger parameters = better results but longer processing time
            """)

if __name__ == "__main__":
    main()