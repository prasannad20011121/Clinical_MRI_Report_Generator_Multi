import os
import io
import time
import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ===================================================
# Google Drive Links for All Models
# ===================================================
MODEL_LINKS = {
    "main": "https://drive.google.com/uc?id=1MrmfGNWW6Msz71WTcrCJcouk5vyDWhMq",
    "brain": "https://drive.google.com/uc?id=1MFRWHTsp830qpVFm19x-74gQ3h6XsJ73",
    "bone": "https://drive.google.com/uc?id=1cFVYwUz8rVY1RBFs4tCJzvZWqZb14WyM",
    "breast": "https://drive.google.com/uc?id=YOUR_BREAST_MODEL_ID",
    "kidney": "https://drive.google.com/uc?id=YOUR_KIDNEY_MODEL_ID"
}

# ===================================================
# Utility: Safe Downloader
# ===================================================
def safe_download(url, path, retries=3):
    """Download model from Google Drive and verify file exists."""
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(path):
                return path
            st.info(f"📥 Downloading {os.path.basename(path)} (attempt {attempt})...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with open(path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total) * 100 if total else 0
                    st.write(f"Downloading... {percent:.1f}%")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                st.success(f"✅ Downloaded {os.path.basename(path)} successfully!")
                return path
        except Exception as e:
            st.warning(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(3)
    st.error(f"❌ Could not download {os.path.basename(path)} after {retries} attempts.")
    return None

# ===================================================
# Load Models (Cached)
# ===================================================
@st.cache_resource
def load_models():
    models = {}
    os.makedirs("models", exist_ok=True)

    for key, url in MODEL_LINKS.items():
        model_path = f"models/{key}_model.keras"
        downloaded = safe_download(url, model_path)
        if downloaded:
            try:
                models[key] = tf.keras.models.load_model(model_path)
                st.success(f"✅ Loaded {key} model successfully.")
            except Exception as e:
                st.error(f"❌ Failed to load {key} model after retry: {e}")
                models[key] = None
        else:
            models[key] = None
    return models

# ===================================================
# PDF Report Generator
# ===================================================
def generate_pdf(patient_name, age, diagnosis, result, image, do_list, dont_list):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "AI Medical Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 100, f"Patient Name: {patient_name}")
    c.drawString(40, height - 120, f"Age: {age}")
    c.drawString(40, height - 140, f"Scan Type: {diagnosis}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 180, "Diagnosis Result:")
    c.setFont("Helvetica", 12)
    text = c.beginText(40, height - 200)
    text.textLines(result)
    c.drawText(text)

    if image:
        image = image.resize((250, 250))
        c.drawImage(ImageReader(image), width - 320, height - 420, width=250, height=250)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 460, "Do's:")
    c.setFont("Helvetica", 12)
    text = c.beginText(60, height - 480)
    text.textLines(do_list)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 540, "Don'ts:")
    c.setFont("Helvetica", 12)
    text = c.beginText(60, height - 560)
    text.textLines(dont_list)
    c.drawText(text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ===================================================
# Streamlit UI
# ===================================================
st.set_page_config(page_title="AI Medical Report Generator", layout="wide")
st.title("🧠 AI Medical Report Generator")

models = load_models()

with st.form("patient_form"):
    patient_name = st.text_input("👤 Patient Name")
    age = st.number_input("🎂 Age", 0, 120, 25)
    diagnosis = st.selectbox("🧪 Select Diagnosis Type", ["Brain", "Bone", "Main", "Breast", "Kidney"])
    uploaded_img = st.file_uploader("📸 Upload MRI / X-Ray Image", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Generate Report")

if submitted:
    if not uploaded_img:
        st.error("Please upload a scan image first.")
    else:
        model_key = diagnosis.lower()
        model = models.get(model_key)

        if model is None:
            st.error(f"Model for {diagnosis} is not loaded properly.")
        else:
            img = Image.open(uploaded_img).convert("RGB").resize((224, 224))
            img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(img_arr)
            pred_class = np.argmax(preds, axis=1)[0]

            result_text = f"Model Prediction: Class {pred_class}\n"
            result_text += "⚕️ Consult your doctor for medical confirmation.\n"

            do_list = "- Follow up with your doctor.\n- Maintain a healthy diet.\n- Take prescribed medication regularly."
            dont_list = "- Do not ignore symptoms.\n- Avoid self-medication.\n- Don’t delay further scans if advised."

            st.subheader("📋 Report Summary")
            st.write(result_text)

            pdf = generate_pdf(patient_name, age, diagnosis, result_text, img, do_list, dont_list)
            st.download_button("⬇️ Download Report as PDF", pdf, file_name=f"{patient_name}_report.pdf")

