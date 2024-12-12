import streamlit as st  
import random
from PIL import Image, ImageOps
import numpy as np
import os , sys , pathlib
from classification_app.data_visualization import inference

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML


with st.sidebar:
        st.image('assets/image_classification.jpg')
        st.title("Image Classification")
        st.subheader("This application is trained to classify amonst 15 classess of vegetables ")

st.write("""
         # Upload a photo to check is robot's will uprise soon or not .......!
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    pred_class , score = inference(file)
  
    st.sidebar.error("Accuracy : " + str(score) + " %")
    st.balloons()
    st.sidebar.success(pred_class)