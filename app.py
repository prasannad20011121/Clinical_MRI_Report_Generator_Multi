import os
import io
import uuid
import datetime
import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


MODEL_LINKS = {
    "main": "https://drive.google.com/file/d/1MrmfGNWW6Msz71WTcrCJcouk5vyDWhMq/view?usp=drive_link",
    "brain": "https://drive.google.com/file/d/1MFRWHTsp830qpVFm19x-74gQ3h6XsJ73/view?usp=drive_link",
    "bone": "https://drive.google.com/file/d/1cFVYwUz8rVqu6gjlMW-_wYoukyCpto5h/view?usp=drive_link",
    "breast": "https://drive.google.com/file/d/1aQ327zLaqHqKrw30qOXlOYPPW3NScFDU/view?usp=drive_link",  
    "kidney": "https://drive.google.com/file/d/1ZAmC8nssodO5IVWhUMpmpOxxdoIg1H2d/view?usp=drive_link"   
}


st.set_page_config(page_title="AI Medical Report Generator", layout="wide")
st.title(" AI Medical Report Generator")
st.caption("Upload a scan → AI Diagnosis → LLM Report → Download PDF")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MAIN_CLASSES = ['bone', 'brain', 'breast', 'kidney']
BRAIN_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BONE_CLASSES = ['fractured', 'not fractured']
BREAST_CLASSES = ['benign', 'malignant']
KIDNEY_CLASSES = ['cyst', 'normal', 'stone', 'tumor']


def download_model_if_missing(name):
    """Download model file if not found locally"""
    local_path = os.path.join(MODEL_DIR, f"{name}_model.keras")
    if not os.path.exists(local_path):
        st.info(f" Downloading {name} model from Drive... (may take a few minutes)")
        url = MODEL_LINKS.get(name)
        if not url:
            st.error(f"No Drive link configured for model '{name}'")
            return None
        response = requests.get(url, stream=True)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        st.success(f"Downloaded {name} model!")
    return local_path

@st.cache_resource
def load_all_models():
    models = {}
    for key in MODEL_LINKS.keys():
        path = download_model_if_missing(key)
        try:
            models[key] = tf.keras.models.load_model(path)
            st.success(f" {key.capitalize()} model loaded")
        except Exception as e:
            st.error(f" Could not load {key} model: {e}")
            models[key] = None
    return models

models = load_all_models()


USE_GEMINI = True
try:
    import google.generativeai as genai
except ImportError:
    USE_GEMINI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if USE_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-2.5-pro")
else:
    llm_model = None
    st.warning("Gemini API not found. Fallback text reports will be used.")


def preprocess_image(img, model):
    """Preprocess image dynamically based on model input shape."""
    img = img.resize((224, 224))
    expected_channels = model.input_shape[-1]

    if expected_channels == 1:
        img = img.convert("L")
        arr = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))
    else:
        img = img.convert("RGB")
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return arr


def predict_main(img_tensor):
    preds = models["main"].predict(img_tensor)
    idx = int(np.argmax(preds))
    return MAIN_CLASSES[idx], float(preds[0][idx])

def predict_domain(organ, img_tensor):
    classes = {
        "brain": BRAIN_CLASSES,
        "bone": BONE_CLASSES,
        "breast": BREAST_CLASSES,
        "kidney": KIDNEY_CLASSES
    }[organ]
    model = models[organ]
    preds = model.predict(img_tensor)
    idx = int(np.argmax(preds))
    return classes[idx], float(preds[0][idx])


def local_report(organ, finding, mode):
    if mode == "Doctor Mode":
        return f"**FINDINGS:** The AI detected {finding} in {organ}. Further imaging recommended."
    else:
        return f"The AI found signs of {finding} in your {organ}. Please consult your doctor."


def generate_pdf(report_text, image, organ, organ_conf, finding_conf):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-40, "AI MEDICAL REPORT")
    c.setFont("Helvetica", 11)
    c.drawString(50, height-70, f"Organ: {organ.capitalize()}")
    c.drawString(50, height-90, f"Organ Confidence: {organ_conf*100:.2f}%")
    c.drawString(50, height-110, f"Finding Confidence: {finding_conf*100:.2f}%")

    image = image.convert("RGB")
    img_buf = io.BytesIO()
    image.save(img_buf, format="PNG")
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, height-400, width=200, preserveAspectRatio=True)

    text_obj = c.beginText(50, height-420)
    text_obj.textLines(report_text)
    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer


col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader(" Upload Image")
    uploaded = st.file_uploader("Upload JPG / PNG", type=["jpg", "jpeg", "png"])
    mode = st.radio("Report Type", ["Doctor Mode", "Patient Mode"], horizontal=True)

with col2:
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Scan", use_column_width=True)

        if st.button(" Generate Report"):
            with st.spinner("Analyzing..."):
                main_input = preprocess_image(img, models["main"])
                organ, conf_org = predict_main(main_input)

                domain_input = preprocess_image(img, models[organ])
                finding, conf_find = predict_domain(organ, domain_input)

                st.success(f"Organ: {organ.upper()} ({conf_org*100:.1f}%) | Finding: {finding.upper()} ({conf_find*100:.1f}%)")

            with st.spinner("Generating Report..."):
                report_text = ""
                if llm_model:
                    try:
                        prompt = f"Generate a detailed {mode} report for organ '{organ}' with finding '{finding}'."
                        response = llm_model.generate_content(prompt)
                        report_text = response.text.strip()
                    except Exception:
                        report_text = local_report(organ, finding, mode)
                else:
                    report_text = local_report(organ, finding, mode)

            st.subheader(" Generated Report")
            st.text_area("Report Text", value=report_text, height=400)

            pdf = generate_pdf(report_text, img, organ, conf_org, conf_find)
            st.download_button("Download PDF", data=pdf, file_name=f"{organ}_report.pdf", mime="application/pdf")

    else:
        st.info("Please upload an image to start diagnosis.")

st.markdown("---")
st.caption(" For educational use only. Consult a medical professional for diagnosis.")

