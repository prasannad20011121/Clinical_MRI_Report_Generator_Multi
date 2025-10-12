# app.py
import os
import io
import uuid
import datetime
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import requests

# ======================
# ✅ Streamlit Config
# ======================
st.set_page_config(page_title="AI Medical Report Generator", layout="wide")
st.title("AI Medical Report Generator")
st.caption("Upload a medical image → AI Diagnosis → LLM-based report → Download professional PDF")

# ======================
# ✅ Gemini Integration
# ======================
USE_GEMINI = True
try:
    import google.generativeai as genai
except ImportError:
    USE_GEMINI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if USE_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-1.5-pro-latest")
else:
    llm_model = None
    st.warning("⚠️ Gemini API key not found. Please add it to your Streamlit Cloud secrets.")

# ======================
# ✅ Model Config
# ======================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

MAIN_CLASSES = ['bone', 'brain', 'breast', 'kidney']
BRAIN_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BONE_CLASSES = ['fractured', 'not fractured']
BREAST_CLASSES = ['benign', 'malignant']
KIDNEY_CLASSES = ['cyst', 'normal', 'stone', 'tumor']

# Google Drive "uc?id=" links for downloading models
MODEL_URLS = {
    "main": "https://drive.google.com/uc?id=1MrmfGNWW6Msz71WTcrCJcouk5vyDWhMq",
    "brain": "https://drive.google.com/uc?id=1MFRWHTsp830qpVFm19x-74gQ3h6XsJ73",
    "bone": "https://drive.google.com/uc?id=1cFVYwUz8rVqu6gjlMW-_wYoukyCpto5h",
    "breast": "https://drive.google.com/uc?id=1aQ327zLaqHqKrw30qOXlOYPPW3NScFDU",
    "kidney": "https://drive.google.com/uc?id=1ZAmC8nssodO5IVWhUMpmpOxxdoIg1H2d"
}

MODEL_PATHS = {k: os.path.join(MODEL_DIR, f"{k}_model.keras") for k in MODEL_URLS.keys()}

# ======================
# ✅ Utility: Download from Google Drive
# ======================
def download_from_drive(url, dest_path):
    """Download model from Google Drive link to destination path."""
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        if token:
            params = {'id': url.split('id=')[1], 'confirm': token}
            response = session.get(url, params=params, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)
        st.success(f"✅ Downloaded: {os.path.basename(dest_path)}")
    except Exception as e:
        st.error(f"Error downloading {dest_path}: {e}")
        raise

# ======================
# ✅ Load All Models (cached)
# ======================
@st.cache_resource
def load_models():
    st.info("Loading AI models from Google Drive (first time only)...")
    loaded = {}
    for key, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.warning(f"Model '{key}' not found locally. Downloading...")
            download_from_drive(MODEL_URLS[key], path)
        loaded[key] = tf.keras.models.load_model(path)
    st.success("✅ All models loaded and cached successfully!")
    return loaded

models = load_models()

# ======================
# ✅ Image Preprocessing & Prediction
# ======================
def preprocess_image(pil_img):
    img = pil_img.convert("L").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

def predict_main(img_tensor):
    preds = models["main"].predict(img_tensor)
    idx = int(np.argmax(preds))
    return MAIN_CLASSES[idx], float(preds[0][idx])

def predict_domain(organ, img_tensor):
    model_domain = models[organ]
    classes = {
        "brain": BRAIN_CLASSES,
        "bone": BONE_CLASSES,
        "breast": BREAST_CLASSES,
        "kidney": KIDNEY_CLASSES
    }[organ]
    preds = model_domain.predict(img_tensor)
    idx = int(np.argmax(preds))
    return classes[idx], float(preds[0][idx])

# ======================
# ✅ Local Report Fallback
# ======================
def local_report(organ, finding, mode):
    today = datetime.datetime.now().strftime("%d-%b-%Y")
    if mode == "Doctor Mode":
        return f"""
**PATIENT:** [Patient Name]
**MRN:** [Medical Record Number]
**DATE OF SERVICE:** {today}
**EXAMINATION:**
AI-assisted {organ.capitalize()} imaging analysis.
**FINDINGS:**
The scan suggests a "{finding}" finding in the {organ}.
**IMPRESSION:**
AI indicates possible {finding}. Recommend clinical correlation.
**RECOMMENDATIONS:**
1. Specialist consultation
2. Confirmatory imaging
3. Follow-up evaluation
"""
    else:
        return f"""
**SUMMARY OF YOUR SCAN**
**Date:** {today}
**Scan Type:** {organ.capitalize()} Scan
**AI Finding:** {finding.capitalize()}
**WHAT THIS MEANS:**
Our AI system detected signs of "{finding.lower()}" in your {organ}.
**NEXT STEPS:**
Consult your doctor for detailed evaluation and next investigations.
**DO'S:**
- Schedule a follow-up with your doctor
- Stay calm and follow medical advice
**DON'TS:**
- Don’t self-diagnose or panic
- Don’t alter medication without medical guidance
"""

# ======================
# ✅ PDF Report Generator
# ======================
def generate_pdf(report_text, image, organ, org_conf, find_conf):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 40 * mm, "AI MEDICAL DIAGNOSTIC REPORT")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, height - 45 * mm, "Generated by CNN + Gemini LLM System")

    today = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    report_id = str(uuid.uuid4())[:8].upper()
    c.drawString(25 * mm, height - 60 * mm, f"Date: {today}")
    c.drawRightString(width - 25 * mm, height - 60 * mm, f"Report ID: {report_id}")

    # Image
    img_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)
    img_reader = ImageReader(img_buf)
    c.drawImage(img_reader, 25 * mm, height - 120 * mm, width=60 * mm, preserveAspectRatio=True)

    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100 * mm, height - 80 * mm, "AI Prediction Summary")
    c.setFont("Helvetica", 10)
    c.drawString(100 * mm, height - 95 * mm, f"Organ: {organ.capitalize()}")
    c.drawString(100 * mm, height - 105 * mm, f"Organ Confidence: {org_conf * 100:.2f}%")
    c.drawString(100 * mm, height - 115 * mm, f"Finding: {finding.capitalize()} ({find_conf * 100:.2f}%)")

    # Report Text
    text = c.beginText(25 * mm, height - 140 * mm)
    text.setFont("Helvetica", 10)
    for line in report_text.split("\n"):
        text.textLine(line.strip())
    c.drawText(text)

    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(width / 2, 25 * mm, "Disclaimer: AI-generated report for research use only.")
    c.save()

    buffer.seek(0)
    return buffer

# ======================
# ✅ Streamlit App UI
# ======================
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Upload Medical Image")
    uploaded_file = st.file_uploader("Upload JPG/PNG medical image", type=["jpg", "jpeg", "png"])
    mode = st.radio("Select Report Mode", ["Doctor Mode", "Patient Mode"], horizontal=True)

with col2:
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Generate Report", type="primary"):
            with st.spinner("Analyzing image..."):
                tensor = preprocess_image(pil_img)
                organ, conf_org = predict_main(tensor)
                finding, conf_find = predict_domain(organ, tensor)
                st.success(f"Organ: {organ.upper()} ({conf_org*100:.1f}%) | Finding: {finding.upper()} ({conf_find*100:.1f}%)")

            with st.spinner("Generating detailed report..."):
                if llm_model:
                    try:
                        prompt = f"""
                        Act as a senior radiologist AI.
                        Generate a full medical report for mode: {mode}.
                        Organ: {organ}, Finding: {finding}.
                        Use structured sections: FINDINGS, IMPRESSION, RECOMMENDATIONS, etc.
                        """
                        response = llm_model.generate_content(prompt)
                        report_text = response.text.strip()
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
                        report_text = local_report(organ, finding, mode)
                else:
                    report_text = local_report(organ, finding, mode)

            st.subheader("Generated Report")
            st.text_area("Medical Report", value=report_text, height=400)

            pdf_data = generate_pdf(report_text, pil_img, organ, conf_org, conf_find)
            st.download_button(
                label="⬇️ Download Full Report (PDF)",
                data=pdf_data,
                file_name=f"{organ}_report_{str(uuid.uuid4())[:4]}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Please upload a medical image to start analysis.")

st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This AI system is for educational and research purposes only.")
