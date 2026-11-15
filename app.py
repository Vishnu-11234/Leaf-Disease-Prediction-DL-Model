
# # # 6. Image preprocessing

# import streamlit as st
# import torch
# import torch.nn as nn
# from pathlib import Path
# from PIL import Image
# from torchvision import transforms

# # ==========================================
# # 1. PAGE CONFIGURATION
# # ==========================================

# st.set_page_config(
#     page_title="PlantCare AI | Disease Detection",
#     page_icon="🌿",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "model" / "plant_disease_cnn.pth"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ==========================================
# # 2. CUSTOM CSS
# # ==========================================

# st.markdown("""
# <style>

# /* Main application */
# .stApp {
#     background-color: #f4f8f4;
# }

# /* Remove default top padding */
# .block-container {
#     padding-top: 2rem;
#     padding-bottom: 3rem;
#     max-width: 1200px;
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: #123c2a;
# }

# [data-testid="stSidebar"] * {
#     color: white !important;
# }

# /* Hero section */
# .hero {
#     background: linear-gradient(120deg, #123c2a, #23834f);
#     padding: 38px;
#     border-radius: 20px;
#     color: white;
#     margin-bottom: 30px;
#     box-shadow: 0 8px 25px rgba(18, 60, 42, 0.15);
# }

# .hero h1 {
#     color: white;
#     font-size: 38px;
#     font-weight: 800;
#     margin-bottom: 8px;
# }

# .hero p {
#     color: #e0f2e7;
#     font-size: 16px;
# }

# /* Section headings */
# .section-title {
#     color: #174b32;
#     font-size: 23px;
#     font-weight: 750;
#     margin-bottom: 15px;
# }

# /* Information cards */
# .info-card {
#     background: white;
#     padding: 22px;
#     border-radius: 15px;
#     border: 1px solid #e2eee5;
#     box-shadow: 0 3px 12px rgba(0,0,0,0.04);
#     min-height: 120px;
# }

# .info-card h3 {
#     color: #174b32;
#     font-size: 17px;
#     margin-bottom: 8px;
# }

# .info-card p {
#     color: #63756a;
#     font-size: 14px;
# }

# /* Upload section */
# [data-testid="stFileUploader"] {
#     background: white;
#     padding: 15px;
#     border-radius: 15px;
#     border: 1px dashed #68a77b;
# }

# /* Buttons */
# .stButton > button {
#     background: linear-gradient(90deg, #176b3a, #239653);
#     color: white;
#     border: none;
#     border-radius: 10px;
#     padding: 12px 25px;
#     font-size: 16px;
#     font-weight: 650;
#     width: 100%;
#     transition: 0.2s;
# }

# .stButton > button:hover {
#     background: #10532e;
#     color: white;
#     transform: translateY(-2px);
# }

# /* Prediction card */
# .prediction-card {
#     background: white;
#     padding: 25px;
#     border-radius: 16px;
#     border-left: 6px solid #239653;
#     box-shadow: 0 4px 18px rgba(0,0,0,0.06);
#     margin-top: 15px;
# }

# .prediction-card h2 {
#     color: #174b32;
#     font-size: 25px;
# }

# .prediction-card p {
#     color: #64766a;
# }

# /* Footer */
# .footer {
#     text-align: center;
#     color: #718078;
#     font-size: 13px;
#     padding: 25px 0 5px;
# }

# </style>
# """, unsafe_allow_html=True)


# # ==========================================
# # 3. CNN ARCHITECTURE
# # ==========================================

# class CNN(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.AdaptiveAvgPool2d((4, 4))
#         )

#         # IMPORTANT:
#         # This classifier is an example.
#         # It MUST match your training notebook exactly.

#         self.fc_layers = nn.Sequential(
#             nn.Linear(128 * 4 * 4, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 38)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.fc_layers(x)
#         return x


# # ==========================================
# # 4. CLASS NAMES
# # ==========================================

# class_names = [
#     "Apple___Apple_scab",
#     "Apple___Black_rot",
#     "Apple___Cedar_apple_rust",
#     "Apple___healthy",
#     "Blueberry___healthy",
#     "Cherry_(including_sour)___Powdery_mildew",
#     "Cherry_(including_sour)___healthy",
#     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
#     "Corn_(maize)___Common_rust_",
#     "Corn_(maize)___Northern_Leaf_Blight",
#     "Corn_(maize)___healthy",
#     "Grape___Black_rot",
#     "Grape___Esca_(Black_Measles)",
#     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
#     "Grape___healthy",
#     "Orange___Haunglongbing_(Citrus_greening)",
#     "Peach___Bacterial_spot",
#     "Peach___healthy",
#     "Pepper,_bell___Bacterial_spot",
#     "Pepper,_bell___healthy",
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Raspberry___healthy",
#     "Soybean___healthy",
#     "Squash___Powdery_mildew",
#     "Strawberry___Leaf_scorch",
#     "Strawberry___healthy",
#     "Tomato___Bacterial_spot",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot",
#     "Tomato___Spider_mites Two-spotted_spider_mite",
#     "Tomato___Target_Spot",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "Tomato___Tomato_mosaic_virus",
#     "Tomato___healthy"
# ]


# # ==========================================
# # 5. LOAD MODEL
# # ==========================================

# @st.cache_resource
# def load_model():

#     # Only use trusted model files.
#     saved_model = torch.load(
#         MODEL_PATH,
#         map_location=device,
#         weights_only=False
#     )

#     saved_model = saved_model.to(device)
#     saved_model.eval()

#     return saved_model


# # ==========================================
# # 6. IMAGE TRANSFORMATION
# # ==========================================

# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         (0.5, 0.5, 0.5),
#         (0.5, 0.5, 0.5)
#     )
# ])


# # ==========================================
# # 7. SIDEBAR
# # ==========================================

# with st.sidebar:

#     st.markdown("# 🌿 PlantCare AI")

#     st.markdown("---")

#     st.markdown("### Navigation")
#     st.markdown("🏠 Disease Detection")
#     st.markdown("🔬 CNN-Based Prediction")
#     st.markdown("📊 Confidence Analysis")

#     st.markdown("---")

#     st.markdown("### About the Model")
#     st.write("Architecture: CNN")
#     st.write("Framework: PyTorch")
#     st.write("Dataset: PlantVillage")
#     st.write("Classes: 38")

#     st.markdown("---")

#     st.markdown("### System Status")

#     if MODEL_PATH.exists():
#         st.success("Model file available")
#     else:
#         st.error("Model file missing")

#     st.caption(f"Running on: {device}")


# # ==========================================
# # 8. HERO SECTION
# # ==========================================

# st.markdown("""
# <div class="hero">

#     <h1>🌱 Plant Disease Detection</h1>

#     <p>
#         Intelligent plant health analysis powered by
#         Convolutional Neural Networks and Deep Learning.
#     </p>

#     <p>
#         Upload a leaf image and let AI analyze its health.
#     </p>

# </div>
# """, unsafe_allow_html=True)


# # ==========================================
# # 9. INFORMATION CARDS
# # ==========================================

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="info-card">
#         <h3>🌿 Smart Detection</h3>
#         <p>Identify plant diseases using a trained CNN model.</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="info-card">
#         <h3>⚡ Fast Prediction</h3>
#         <p>Analyze uploaded leaf images with deep learning.</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="info-card">
#         <h3>📊 Confidence Analysis</h3>
#         <p>View prediction confidence and alternative results.</p>
#     </div>
#     """, unsafe_allow_html=True)


# st.markdown("---")


# # ==========================================
# # 10. LOAD MODEL
# # ==========================================

# if not MODEL_PATH.exists():

#     st.error(f"Model file not found: {MODEL_PATH}")
#     st.stop()

# try:
#     model = load_model()

# except Exception as e:

#     st.error(f"Unable to load model: {e}")
#     st.stop()


# # ==========================================
# # 11. IMAGE UPLOAD SECTION
# # ==========================================

# st.markdown(
#     '<div class="section-title">📸 Upload Your Plant Image</div>',
#     unsafe_allow_html=True
# )

# left, right = st.columns([1, 1], gap="large")

# with left:

#     uploaded_file = st.file_uploader(
#         "Choose a leaf image",
#         type=["jpg", "jpeg", "png"],
#         help="Upload a clear image of a plant leaf."
#     )

#     st.caption("Supported formats: JPG, JPEG, PNG")

#     predict_button = st.button(
#         "🔍 Analyze Plant Health",
#         use_container_width=True
#     )


# with right:

#     if uploaded_file is not None:

#         image = Image.open(uploaded_file).convert("RGB")

#         st.image(
#             image,
#             caption="Uploaded Plant Leaf",
#             use_container_width=True
#         )

#     else:

#         st.markdown("""
#         <div class="info-card" style="text-align:center; padding:50px 20px;">

#             <h1>🍃</h1>

#             <h3>Your plant image will appear here</h3>

#             <p>Upload an image to begin disease detection.</p>

#         </div>
#         """, unsafe_allow_html=True)


# # ==========================================
# # 12. PREDICTION
# # ==========================================

# if predict_button:

#     if uploaded_file is None:

#         st.warning("Please upload a plant image first.")

#     else:

#         with st.spinner("🧠 AI is analyzing your plant image..."):

#             try:

#                 input_tensor = transform(image)
#                 input_tensor = input_tensor.unsqueeze(0).to(device)

#                 with torch.inference_mode():

#                     outputs = model(input_tensor)

#                     probabilities = torch.softmax(outputs, dim=1)

#                     top_probabilities, top_indices = torch.topk(
#                         probabilities,
#                         k=3,
#                         dim=1
#                     )

#                 predicted_index = top_indices[0][0].item()

#                 if len(class_names) != outputs.shape[1]:
#                     st.error(
#                         "Mismatch between model output classes "
#                         "and class_names."
#                     )
#                     st.stop()

#                 predicted_class = class_names[predicted_index]

#                 confidence = top_probabilities[0][0].item() * 100

#                 is_healthy = "healthy" in predicted_class.lower()

#                 # Format prediction name

#                 display_name = predicted_class.replace("___", " - ")
#                 display_name = display_name.replace("_", " ")

#                 st.markdown("---")

#                 st.markdown(
#                     '<div class="section-title">📊 Analysis Results</div>',
#                     unsafe_allow_html=True
#                 )

#                 if is_healthy:

#                     st.success("🌿 Healthy Plant Detected")

#                     border_color = "#239653"
#                     status = "Healthy"

#                 else:

#                     st.warning("⚠️ Potential Disease Detected")

#                     border_color = "#d97706"
#                     status = "Disease Detected"

#                 # Prediction card

#                 st.markdown(f"""
#                 <div class="prediction-card"
#                      style="border-left-color:{border_color};">

#                     <p>DETECTION STATUS: {status}</p>

#                     <h2>{display_name}</h2>

#                     <p>Predicted plant condition based on CNN analysis.</p>

#                 </div>
#                 """, unsafe_allow_html=True)

#                 # Metrics

#                 metric1, metric2 = st.columns(2)

#                 with metric1:

#                     st.metric(
#                         "Prediction Confidence",
#                         f"{confidence:.2f}%"
#                     )

#                 with metric2:

#                     st.metric(
#                         "Predicted Class",
#                         f"{predicted_index + 1} / {len(class_names)}"
#                     )

#                 st.markdown("### Confidence Score")

#                 st.progress(confidence / 100)

#                 # Top 3 predictions

#                 st.markdown("### 🏆 Top 3 Predictions")

#                 for rank in range(3):

#                     idx = top_indices[0][rank].item()
#                     prob = top_probabilities[0][rank].item() * 100

#                     name = class_names[idx]
#                     name = name.replace("___", " - ").replace("_", " ")

#                     st.write(f"**{rank + 1}. {name}**")

#                     st.progress(prob / 100)

#                     st.caption(f"Confidence: {prob:.2f}%")

#                 st.info(
#                     "AI predictions are advisory, not a confirmed "
#                     "agricultural diagnosis. Check the plant under "
#                     "appropriate conditions before taking action."
#                 )

#             except Exception as e:

#                 st.error(f"Prediction failed: {e}")


# # ==========================================
# # 13. FOOTER
# # ==========================================

# st.markdown("---")

# st.markdown("""
# <div class="footer">

#     <h3>🌿 PlantCare AI</h3>

#     <p>
#         Powered by PyTorch | CNN | Deep Learning
#     </p>

#     <p>
#         Building smarter agriculture through Artificial Intelligence.
#     </p>

# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
import io


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. STREAMLIT THEME STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f8f5;
}

[data-testid="stSidebar"] {
    background-color: #173e2b;
}

[data-testid="stSidebar"] * {
    color: white;
}

.stButton > button {
    background-color: #21834d;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #145c35;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #dce9df;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. CNN CLASS
# ==========================================

# This version assumes you saved the COMPLETE MODEL
# using torch.save(model, MODEL_PATH).
#
# During unpickling, PyTorch restores the saved layers.
# Keep the forward method consistent with your training CNN.

class CNN(nn.Module):

    def forward(self, x):

        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        return x


# ==========================================
# 4. PLANTVILLAGE CLASS NAMES
# ==========================================

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ==========================================
# 5. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.to(device)
    model.eval()

    return model


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

])


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌿 PlantCare AI")

    st.caption("Intelligent Agriculture System")

    st.divider()

    st.subheader("Navigation")

    st.markdown("🏠 Disease Detection")
    st.markdown("🔬 CNN Prediction")
    st.markdown("📊 Confidence Analysis")

    st.divider()

    st.subheader("Model Information")

    st.write("**Architecture:** CNN")
    st.write("**Framework:** PyTorch")
    st.write("**Dataset:** PlantVillage")
    st.write("**Classes:** 38")

    st.divider()

    st.subheader("System Information")

    st.success(f"Device: {device}")

    if MODEL_PATH.exists():
        st.success("Model file available")
    else:
        st.error("Model file missing")


# ==========================================
# 8. MAIN HEADER
# ==========================================

st.title("🌱 Plant Disease Detection")

st.markdown(
    "### AI-powered plant health monitoring system"
)

st.write(
    "Upload a plant leaf image and let our deep learning "
    "model identify its potential disease."
)

st.divider()


# ==========================================
# 9. FEATURE CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    with st.container(border=True):

        st.subheader("🌿 Smart Detection")

        st.write(
            "Identify plant diseases using "
            "Convolutional Neural Networks."
        )

with col2:

    with st.container(border=True):

        st.subheader("⚡ Fast Analysis")

        st.write(
            "Analyze leaf images and generate "
            "predictions using deep learning."
        )

with col3:

    with st.container(border=True):

        st.subheader("📊 Confidence Score")

        st.write(
            "Explore prediction probabilities "
            "and alternative disease classes."
        )


st.divider()


# ==========================================
# 10. LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():

    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

try:

    model = load_model()

except Exception as e:

    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# 11. IMAGE UPLOAD
# ==========================================

st.header("📸 Upload Plant Image")

left, right = st.columns([1, 1], gap="large")

uploaded_file = None
image = None

with left:

    with st.container(border=True):

        st.subheader("Choose Your Image")

        uploaded_file = st.file_uploader(
            "Upload a plant leaf image",
            type=["jpg", "jpeg", "png"]
        )

        st.caption(
            "Supported formats: JPG, JPEG and PNG"
        )

        predict_button = st.button(
            "🔍 Analyze Plant Health",
            use_container_width=True
        )


with right:

    with st.container(border=True):

        st.subheader("Image Preview")

        if uploaded_file is not None:

            image = Image.open(
                io.BytesIO(uploaded_file.getvalue())
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Plant Leaf",
                use_container_width=True
            )

        else:

            st.markdown("### 🍃")

            st.info(
                "Your uploaded leaf image will appear here."
            )


# ==========================================
# 12. PREDICTION
# ==========================================

if predict_button:

    if uploaded_file is None:

        st.warning("Please upload an image first.")

    else:

        try:

            with st.spinner(
                "🧠 AI is analyzing your plant image..."
            ):

                input_tensor = transform(image)

                input_tensor = (
                    input_tensor.unsqueeze(0).to(device)
                )

                with torch.inference_mode():

                    outputs = model(input_tensor)

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    if outputs.shape[1] != len(class_names):
                        st.error(
                            "Model output does not match "
                            "the 38 class names."
                        )
                        st.stop()

                    top_probabilities, top_indices = torch.topk(
                        probabilities,
                        k=min(3, len(class_names)),
                        dim=1
                    )

            # Main prediction

            predicted_index = top_indices[0][0].item()

            predicted_class = class_names[predicted_index]

            confidence = (
                top_probabilities[0][0].item() * 100
            )

            display_name = predicted_class.replace(
                "___", " - "
            ).replace("_", " ")

            is_healthy = "healthy" in predicted_class.lower()

            # ==================================
            # 13. RESULT DISPLAY
            # ==================================

            st.divider()

            st.header("📊 Prediction Results")

            if is_healthy:

                st.success("🌿 Healthy Plant Detected!")

                with st.container(border=True):

                    st.subheader("Plant Health Status")

                    st.title("✅ Healthy")

                    st.write(display_name)

            else:

                st.warning("⚠️ Potential Disease Detected")

                with st.container(border=True):

                    st.subheader("Detected Condition")

                    st.title("🔬 Disease Detected")

                    st.write(display_name)

            # ==================================
            # 14. METRICS
            # ==================================

            st.subheader("Prediction Overview")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with metric2:

                st.metric(
                    "Predicted Class",
                    f"{predicted_index + 1}/38"
                )

            with metric3:

                st.metric(
                    "Device",
                    str(device).upper()
                )

            # ==================================
            # 15. CONFIDENCE BAR
            # ==================================

            st.subheader("🎯 Prediction Confidence")

            st.progress(confidence / 100)

            st.caption(
                f"Model output probability: {confidence:.2f}%"
            )

            # ==================================
            # 16. TOP 3 PREDICTIONS
            # ==================================

            st.divider()

            st.subheader("🏆 Top 3 Predictions")

            for rank in range(min(3, len(class_names))):

                idx = top_indices[0][rank].item()

                prob = (
                    top_probabilities[0][rank].item() * 100
                )

                name = class_names[idx]

                name = name.replace(
                    "___", " - "
                ).replace("_", " ")

                with st.container(border=True):

                    st.write(
                        f"**{rank + 1}. {name}**"
                    )

                    st.progress(prob / 100)

                    st.caption(
                        f"Confidence: {prob:.2f}%"
                    )

            # ==================================
            # 17. DISCLAIMER
            # ==================================

            st.info(
                "Note: AI predictions are advisory and "
                "should not replace expert agricultural "
                "diagnosis. Model confidence does not "
                "guarantee prediction accuracy."
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.markdown(
    "### 🌿 PlantCare AI"
)

st.caption(
    "Powered by PyTorch | CNN | Deep Learning"
)

st.caption(
    "Building smarter agriculture through Artificial Intelligence."
)
