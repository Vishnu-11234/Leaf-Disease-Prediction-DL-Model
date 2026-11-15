# 🌿 Plant Disease Detection Using Deep Learning

An AI-powered Plant Disease Detection System that uses **Convolutional Neural Networks (CNN)** to identify plant diseases from leaf images. The project uses the PlantVillage dataset and PyTorch to classify plant leaves into 38 different categories, including healthy and diseased conditions.

The application provides an interactive interface built with Streamlit, allowing users to upload leaf images and receive disease predictions along with model confidence scores.

---

## 🚀 Project Overview

Plant diseases can significantly affect agricultural productivity and crop quality. Early identification helps farmers and agricultural researchers take preventive measures.

This project applies Deep Learning and Computer Vision to automatically identify plant diseases from leaf images.

Users can upload a plant leaf image through the Streamlit application, and the trained CNN model analyzes the image to predict its corresponding disease category.

## ✨ Features

- **Image-Based Disease Detection:** Upload plant leaf images for disease classification.
- **CNN-Based Classification:** Uses a Convolutional Neural Network implemented with PyTorch.
- **38-Class Prediction:** Classifies images into 38 PlantVillage categories.
- **Image Preprocessing:** Resizing, tensor conversion, and normalization.
- **Confidence Analysis:** Displays prediction confidence scores.
- **Top 3 Predictions:** Shows the three most probable disease categories.
- **Interactive Web Interface:** User-friendly application developed using Streamlit.
- **GPU Support:** Automatically detects CUDA availability and uses GPU when available.

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) for image classification.

### CNN Architecture

| Layer | Configuration |
|---|---|
| Input | RGB image (128 × 128) |
| Convolution 1 | 3 → 32 filters |
| Activation | ReLU |
| Pooling | Max Pooling (2 × 2) |
| Convolution 2 | 32 → 64 filters |
| Activation | ReLU |
| Pooling | Max Pooling (2 × 2) |
| Convolution 3 | 64 → 128 filters |
| Activation | ReLU |
| Pooling | Max Pooling (2 × 2) |
| Adaptive Pooling | 4 × 4 |
| Classifier | As implemented in the training notebook |
| Output | 38 classes |

### Image Preprocessing

Images are processed using Torchvision transforms:

1. Resize images to 128 × 128 pixels.
2. Convert images into PyTorch tensors.
3. Normalize RGB channels using mean and standard deviation of 0.5.

## 📂 Dataset

**PlantVillage Dataset**

The project uses the PlantVillage dataset, which contains images of healthy and diseased plant leaves.

- Number of classes: 38
- Image type: RGB leaf images
- Classification type: Multiclass image classification

The dataset covers multiple crops, including:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

The dataset includes both healthy leaves and leaves affected by different diseases.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning model development |
| Torchvision | Image preprocessing |
| CNN | Image classification |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pillow (PIL) | Image loading and processing |
| Jupyter Notebook | Model development and experimentation |

## 📁 Project Structure

```text
Leaf_Disease_Prediction/
│
├── model/
│   └── plant_disease_cnn.pth
│
├── notebooks/
│   ├── Cnn.ipynb
│   ├── plantvillage dataset/
│   └── plantvillage_split/
│
├── .virtual_documents/
│
├── anaconda_projects/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── plantvillage_split.zip
```

*Note: The folder structure above reflects the project layout. The model file should be placed inside the `model` directory.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd Leaf_Disease_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📸 Application Workflow

1. Launch the Streamlit application.
2. Upload a plant leaf image.
3. The image is resized and normalized.
4. The processed image is passed to the trained CNN model.
5. The model generates class predictions.
6. Softmax converts model outputs into probability scores.
7. The application displays the predicted disease, confidence score, and top 3 predictions.

## 🔬 Model Inference

The application uses PyTorch inference mode to perform predictions without calculating gradients.

```python
with torch.inference_mode():
    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_index = probabilities.max(
        dim=1
    )
```

The predicted class index is mapped to the corresponding PlantVillage category.

## 🔮 Future Improvements

- Add disease descriptions and symptom information.
- Integrate disease prevention and treatment recommendations.
- Improve model performance through architecture experimentation.
- Add prediction history and downloadable reports.
- Deploy the application to a cloud platform.
- Evaluate model generalization using real-world leaf images.

## ⚠️ Limitations

- Predictions depend on the quality and characteristics of uploaded images.
- Performance on real-world images may differ from PlantVillage dataset performance.
- Model confidence does not guarantee prediction correctness.
- The system is intended as a decision-support tool and not a replacement for expert agricultural diagnosis.

## 👨‍💻 Author

**Vishnu Jante**

- GitHub: [Vishnu-11234](https://github.com/Vishnu-11234)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/vishnu-jante-)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, PyTorch, CNN and Streamlit.** 🌱