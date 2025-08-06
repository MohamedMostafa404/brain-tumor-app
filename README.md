# 🧠 Brain Tumor Detection & Segmentation App

A Streamlit web app for detecting and segmenting brain tumors from MRI scans using deep learning.

---

## 🔬 Notebooks for Model Training

| Notebook | Description |
|----------|-------------|
| `1-Classification-Training.ipynb` | Trains a ResNet-based model to classify MRI scans as Tumor or No Tumor |
| `2-Segmentation-Training.ipynb`   | Trains a U-Net model to perform tumor segmentation on MRI scans |

---

## 🎯 Features

- 🧠 **Classification**: Detects if the MRI image contains a brain tumor.
- 🎨 **Segmentation**: If a tumor exists, shows a segmented mask and an overlay.
- 📤 Easy image upload interface via Streamlit.
- 📈 Trained using high-performance models and optimized datasets.

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow & Keras
- Streamlit
- OpenCV
- NumPy, Matplotlib

---

## 📁 Folder Structure

brain_tumor_app/

├── app.py                          # Streamlit app combining classification & segmentation

├── Model_3.h5                      # Trained classification model (ResNet)

├── Tumor_Segmentation2.h5         # Trained segmentation model (U-Net)

├── 1-Classification-Training.ipynb # Notebook for training the classifier

├── 2-Segmentation-Training.ipynb   # Notebook for training the segmenter

├── requirements.txt               # Python dependencies

└── README.md                      # Project overview and usage instructions

---

## 📦 Requirements
The requirements.txt file includes:

txt

Copy

Edit

streamlit

tensorflow

numpy

opencv-python

Pillow

Make sure you have Python 3.7+ installed.


----------

## 🚀 How to Run the App

1. Install the requirements:
```bash
pip install -r requirements.txt


---------------


## 🚀 Run the Streamlit App

To run the app locally, follow these steps:

1. **Install the required packages** (preferably in a virtual environment):

```bash
pip install -r requirements.txt

------------------


## 📸 Demo (Optional – Add Screenshot or GIF)
You can upload a screenshot or gif of the app here.

---


##👨‍💻 Author
Developed by [Mohamed Mostafa]
Feel free to contribute or suggest improvements!
