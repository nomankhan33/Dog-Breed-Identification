import os
from matplotlib import pyplot as plt
import preprocessor
import streamlit as st
import numpy as np
import tempfile


st.sidebar.title("Dog Breed Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a dog image")
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    custom_path = temp_dir + "/"
    if st.sidebar.button("Predict"):
        loaded_full_model = preprocessor.load_model("20240717-06311721197913-full-image-set-mobilenetv2-Adam.h5")
        custom_image_path = [custom_path + fname for fname in os.listdir(custom_path)]
        custom_data = preprocessor.create_data_batches(
            custom_image_path, test_data=True
        )
        custom_preds = loaded_full_model.predict(custom_data)
        custom_preds_labels = [
            preprocessor.get_pred_label(custom_preds[i])
            for i in range(len(custom_preds))
        ]
        st.title("Prediction")
        custom_images = []
        for image in custom_data.unbatch().as_numpy_iterator():
            custom_images.append(image)

        num_images = len(custom_images)

        fig, axes = plt.subplots(1, num_images, figsize=(18, 10))

        if num_images == 1:
            axes = [axes]

        for i, (image, ax) in enumerate(zip(custom_images, axes)):
            ax.imshow(image)
            ax.set_title("Uploaded Image", fontsize=20)
            ax.axis("off")
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<h1 style="color: grey;">Predicted Breed</h1>',
                unsafe_allow_html=True,
            )
            st.header(custom_preds_labels[0])

        with col2:
            st.markdown(
                '<h1 style="color: grey;">Confidence</h1>',
                unsafe_allow_html=True,
            )
            max_value = np.max(custom_preds) * 100
            percentage_str = f"{max_value:.2f}%"
            st.header(percentage_str)
