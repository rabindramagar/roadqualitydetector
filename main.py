import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from road_quality_classifier import RoadQualityClassifier
from image_utils import predict_class, draw_prediction
import random

# Setting Streamlit page configuration for a better layout
st.set_page_config(
    page_title="Road Quality Detection",
    page_icon="🏗️",
    layout="wide"
)

# Styling the main title with CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        text-align: center;
        color: #1e88e5;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Road Quality Detection 🛣️🔍📝</h1>', unsafe_allow_html=True)

# File uploader with custom styling
uploaded_file = st.file_uploader(
    "Choose an image file to analyze",
    type=['jpg', 'png', 'jpeg'],
    help="Upload your image here to analyze the road quality.",
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        # Converting the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
        # Layout for displaying original and result images
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.header("Original Image")
            st.image(opencv_image, channels="BGR", use_column_width=True)
        
        with col2:
            st.header("Analyzed Image")
            test_img = Image.open(uploaded_file)
            road_quality_class = predict_class(uploaded_file)
            result_img = draw_prediction(test_img, road_quality_class)
            st.image(result_img, use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing your image: {e}", icon="🚨")

else:
    st.subheader("Explore Sample Images")
    
    image_deck = [
        "test_image/autobahn.jpg",
        "test_image/bad_road.jpg",
        "test_image/sample5.jpg",
        "test_image/sample1.jpg"
    ]
    
    # Button styling for a more interactive look
    if st.button('Try a Random Sample Image', help='Click to see analysis on a random sample image'):
        image_number = random.randint(0, len(image_deck) - 1) 
        sample_image_path = image_deck[image_number]
        sample_image = Image.open(sample_image_path)

        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.header("Sample Image")
            st.image(sample_image, channels="RGB", use_column_width=True)

        with col2:
            st.header("Analysis Result")
            road_quality_class = predict_class(sample_image_path)
            result_img = draw_prediction(sample_image, road_quality_class)
            st.image(result_img, use_column_width=True)
        
    # Displaying thumbnails of sample images for quick selection
    st.markdown("### Sample Image Gallery")
    cols = st.columns(4)
    for i, img_path in enumerate(image_deck):
        with cols[i % 4]:
            img = Image.open(img_path)
            st.image(img, width=150, channels="BGR", caption=f"Sample {i+1}")

# Adding some footer or additional information
st.markdown("""
    <style>
    .footer {
        text-align: center;
        color: #757575;
        font-size: 12px;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Made with ❤️ by ~Rabindra Magar
    </div>
    """, unsafe_allow_html=True)