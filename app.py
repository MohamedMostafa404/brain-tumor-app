import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2 as cv
import streamlit as st
import numpy as np

st.set_page_config(page_title="🧠 Brain Tumor App", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection & Segmentation</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        detection_model = load_model('brain_tumor_classifier.keras')
        segmentation_model = load_model('brain_tumor_segmentation_model.keras')
        return detection_model, segmentation_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

Brain_Detection, Brain_Segmentation = load_models()
Brain_Detection_classes = ['No', 'Yes']

if Brain_Detection is None or Brain_Segmentation is None:
    st.error("Models could not be loaded. Please check if the model files exist.")
    st.stop()

# Centered layout using columns
col_center = st.columns([1, 2, 1])[1]

with col_center:
    uploaded_file = st.file_uploader("📤 Upload a brain MRI image", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_file:
        # Show preview (smaller image)
        st.markdown("### 🖼️ Uploaded Image Preview")
        st.image(uploaded_file, width=250)

    # Buttons in same row, centered
    col1, col2 = st.columns([1, 1])
    with col1:
        detection_button = st.button('🔍 Brain Detection', use_container_width=True)
    with col2:
        segmentation_button = st.button('🧩 Brain Segmentation', use_container_width=True)

    # Detection Logic
    if detection_button and uploaded_file:
        try:
            img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_normalized = np.expand_dims(img_array, axis=0) / 255.0

            with st.spinner('Analyzing image...'):
                predictions = Brain_Detection.predict(img_normalized, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            st.markdown("### 🧠 Detection Results")
            if Brain_Detection_classes[predicted_class] == 'No':
                st.success(f'✅ No tumor detected! (Confidence: {confidence:.2f}%)')
            else:
                st.error(f'🚨 Tumor detected! (Confidence: {confidence:.2f}%)')

        except Exception as e:
            st.error(f"Error during detection: {str(e)}")

    # Segmentation Logic
    if segmentation_button and uploaded_file:
        try:
            uploaded_file.seek(0)
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

            if img is None:
                st.error("Could not decode the uploaded image. Please try a different image.")
            else:
                img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_resized = cv.resize(img_rgb, (256, 256))
                img_input = img_resized / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                with st.spinner('Segmenting tumor...'):
                    pred = Brain_Segmentation.predict(img_input, verbose=0)[0]

                pred_mask = (pred > 0.5).astype(np.uint8)

                st.markdown("### 🧩 Segmentation Results")
                if pred_mask.max() > 0:
                    st.success("Tumor region detected.")
                    # Apply mask visualization
                    mask_resized = cv.resize(pred_mask * 255, (img.shape[1], img.shape[0]))
                    overlay = cv.addWeighted(img_rgb, 0.7, cv.cvtColor(mask_resized, cv.COLOR_GRAY2RGB), 0.3, 0)
                    # إعداد الأعمدة الثلاثة
                    col1, col2, col3 = st.columns(3)

                    with col1:
                         st.image(img_rgb, caption="🖼️ Original Image", use_container_width=True)
 
                    with col2:
                         st.image(mask_resized, caption="🧠 Tumor Mask",  use_container_width=True, clamp=True)

                    with col3:
                         st.image(overlay, caption="🔍 Overlay",  use_container_width=True)

                else:
                    st.info("No tumor region detected.")
        except Exception as e:
            st.error(f"Error during segmentation: {str(e)}")
