import streamlit as st
import numpy as np
import pandas as pd
from config import config
import requests
import json
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


def get_explanations(file, model_name):
    files = {'upload_file': file}
    data = {'model_name': model_name}
    res = requests.post(config['API']['URL'] + '/explain', data=data, files=files)
    data = []
    if res.status_code == 200:
        data = res.json()
    return data


def get_benchmarks(file, model_name):
    files = {'upload_file': file}
    data = {'model_name': model_name}
    res = requests.post(config['API']['URL'] + '/benchmarks', data=data, files=files)
    data = []
    if res.status_code == 200:
        data = res.json()
    return data


# ---- set page layout ----
st.set_page_config(
    page_title="Explainable AI in Deep Learning Models for Computer Vision",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Explainable AI in Deep Learning Models for Computer Vision ")

# ---- Navigation bar ---- 
selected = option_menu(None, ["Home", "Benchmark", "Creations"],
                       icons=['house', 'cast', 'command'],
                       menu_icon="cast", default_index=0, orientation="horizontal")
# selected

# ---- Home Page Function ----
if selected == 'Home':
    st.markdown("<h2 style='text-align: center; color: grey;margin-bottom:5%'>ABOUT THE ACTION LEARNING PROJECT</h1>", unsafe_allow_html=True)
    st.markdown("<div style ='display:block;background:#8080801c;padding: 25px 15px 25px 15px;'><ul><li>The software is created using Streamlit and FastApi</li><li>In the Benchmark tab you can use 2 pre-trained models which are VGG16 and Inception and see their explanations with a LIME package</li><li>In the Creation tab we have created our own explainers to eplain how the model in classifying the image</li><li>In 'Benchmark' and 'Creation' tab, the results include classification score and explainer's heatmap </li></ul></div>", unsafe_allow_html=True)
    st.markdown("<h6 style='margin-top:5%'><strong>Professors</strong> : Prof. Bill MANOS and Prof. Alaa BHAKTI</h6>", unsafe_allow_html=True)
    st.markdown("<h6><strong>Contributors</strong> : Rahul JAIKISHEN, Utsav PANDEY and Tanmay MONDKAR</h6>", unsafe_allow_html=True)



# ---- Benchmark Function ----
def benchmark():
    st.subheader("Upload an Image file")
    col1, col2 = st.columns(2)
    with col1:
        models_list = ["VGG16", "Inception"]
        network = st.selectbox("Select the Model", models_list)
    with col2:
        uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        clear_benchmark = st.button("Clear Screen")
    with col2:
        check_benchmark = st.button("Submit")

    if check_benchmark:
        with st.spinner('Leaving the realm of frontend. Hoping onto the API to get the explanations. Waiting for Mr. API to send us back the results ...'):
            pred = get_benchmarks(uploaded_file.getvalue(), network)
        val = json.loads(pred)
        st.subheader(f"Top Classification from {network}")
        values = [network, val['prediction'], val['score']]

        df = pd.DataFrame([values], columns=['Model Name', 'Image Classification', 'Score'])

        st.table(df)
        # st.write(df)

        bytes_data = uploaded_file.read()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(bytes_data, width=450)

        with col2:
            st.subheader("Lime Explainer")
            heatmap = np.array(val['heatmap_lime'])
            fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
            ax.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
            # ax.colorbar()
            ax.axis('off')
            st.write(fig)

        if clear_benchmark:
            st.stop()


# ---- Own Created Explainer Function ----
def own_creation():
    st.subheader("Upload an Image file")
    col1, col2 = st.columns(2)
    with col1:
        models_list = ["VGG16", "Inception"]
        network = st.selectbox("Select the Model", models_list)
    with col2:
        uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    with col1:
        clear_creation = st.button("Clear Screen")
    with col2:
        check_creation = st.button("Submit")

    if check_creation:
        with st.spinner('Leaving the realm of frontend. Hoping onto the API to get the explanations. Waiting for Mr. API to send us back the results ...'):
            pred = get_explanations(uploaded_file.getvalue(), network)
        val = json.loads(pred)
        st.subheader(f"Top Classification from {network}")
        # st.write(val['score'], val['prediction'], column=('score', 'classification'))
        values = [network, val['prediction'], val['score']]

        df = pd.DataFrame([values], columns=['Model Name', 'Image Classification', 'Score'])

        st.table(df)
        bytes_data = uploaded_file.read()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(bytes_data)
        with col2:
            st.subheader("Custom Explainer 1")
            heatmap_custom_2 = np.array(val['explanation_custom_2'])
            fig, ax = plt.subplots()
            ax.imshow(heatmap_custom_2, cmap='RdBu', vmin=-heatmap_custom_2.max(), vmax=heatmap_custom_2.max())
            # fig.colorbar(fig)
            ax.axis('off')
            st.write(fig)

        with col1:
            st.subheader("Original Image")
            st.image(bytes_data)
        with col2:
            st.subheader("Custom Explainer 2")
            # heatmap = np.array(val['heatmap_lime'])
            heatmap_custom = np.array(val['cie_inspiriation'])
            fig, ax = plt.subplots()
            ax.imshow(heatmap_custom, cmap='RdBu', vmin=-heatmap_custom.max(), vmax=heatmap_custom.max())
            # fig.colorbar(fig)
            ax.axis('off')
            st.write(fig)

        if clear_creation:
            st.stop()


# Navigating to pages
if selected == 'Benchmark':
    benchmark()

if selected == 'Creations':
    own_creation()
