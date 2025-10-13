# app.py
import os
import io
import uuid
import datetime
import shutil
import tempfile
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
# ✅ Gemini Integration (supports env or Streamlit secrets)
# ======================
USE_GEMINI = True
try:
    import google.generativeai as genai
except Exception:
    USE_GEMINI = False

# prefer Streamlit secrets (on Streamlit Cloud) then environment variable
GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if USE_GEMINI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    except Exception as e:
        llm_model = None
        st.warning(f"⚠️ Gemini init failed: {e}")
else:
    llm_model = None
    st.warning("⚠️ Gemini API key not found or library missing. LLM fallback will be used.")

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

# Google Drive "uc?id=" links for downloading models (ensure these are "Anyone with link" shareable)
MODEL_URLS = {
    "main": "https://drive.google.com/uc?id=1MrmfGNWW6Msz71WTcrCJcouk5vyDWhMq",
    "brain": "https://drive.google.com/uc?id=1MFRWHTsp830qpVFm19x-74gQ3h6XsJ73",
    "bone": "https://drive.google.com/uc?id=1cFVYwUz8rVqu6gjlMW-_wYoukyCpto5h",
    "breast": "https://drive.google.com/uc?id=1aQ327zLaqHqKrw30qOXlOYPPW3NScFDU",
    "kidney": "https://drive.google.com/uc?id=1ZAmC8nssodO5IVWhUMpmpOxxdoIg1H2d"
}

MODEL_PATHS = {k: os.path.join(MODEL_DIR, f"{k}_model.keras") for k in MODEL_URLS.keys()}

# log file to help debug in Streamlit Cloud
LOG_PATH = os.path.join(MODEL_DIR, "model_load_errors.log")

def log_msg(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} - {msg}\n")

# ======================
# ✅ Utility: Download from Google Drive with progress & safe write
# ======================
def download_from_drive(url, dest_path, st_container=None):
    """
    Robust download from a Google Drive `uc?id=` URL.
    Writes to a temp file first then moves to dest_path to avoid partial files.
    Shows progress in Streamlit if st_container provided.
    """
    try:
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000:
            # already present and reasonably large
            return dest_path

        session = requests.Session()
        response = session.get(url, stream=True)
        # handle Drive confirm token if present
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token:
            params = {"id": url.split("id=")[1], "confirm": token}
            response = session.get(url, params=params, stream=True)

        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp")
        os.close(tmp_fd)
        bytes_written = 0

        if st_container:
            progress_bar = st_container.progress(0)
            status_text = st_container.empty()
        else:
            progress_bar = None
            status_text = None

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
                    if total and progress_bar:
                        progress_bar.progress(min(100, int(bytes_written * 100 / total)))
                        status_text.text(f"Downloaded {bytes_written // 1024} KB of {total // 1024} KB")
        # final move
        shutil.move(tmp_path, dest_path)
        if progress_bar:
            progress_bar.progress(100)
            status_text.text("Download complete.")
        st.success(f"✅ Downloaded: {os.path.basename(dest_path)}")
        return dest_path

    except Exception as e:
        log_msg(f"Download error for {dest_path}: {e}")
        # clean up temp if exists
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

# ======================
# ✅ Load All Models (Safe & Cached)
# ======================
@st.cache_resource
def load_models():
    st.info("🔄 Loading AI models from Google Drive (first time only)...")
    loaded = {}
    # we create a section container so progress messages for downloads appear neatly
    container = st.container()

    for key, path in MODEL_PATHS.items():
        try:
            # If file missing or too small, download
            if not os.path.exists(path) or os.path.getsize(path) < 100_000:
                container.warning(f"Model '{key}' not found or incomplete. Downloading...")
                download_from_drive(MODEL_URLS[key], path, st_container=container)

            container.write(f"📦 Loading {key} model...")
            # prefer compile=False to avoid issues with custom objects
            try:
                loaded_model = tf.keras.models.load_model(path, compile=False)
            except Exception:
                # try again without compile arg (older TF versions)
                loaded_model = tf.keras.models.load_model(path)
            loaded[key] = loaded_model
            container.success(f"✅ Loaded: {key}")

        except Exception as e:
            err_msg = f"Failed to load {key} model: {e}"
            st.error(f"❌ {err_msg}")
            log_msg(err_msg)

            # Retry download once
            try:
                container.warning(f"Retrying download for {key}...")
                download_from_drive(MODEL_URLS[key], path, st_container=container)
                try:
                    loaded_model = tf.keras.models.load_model(path, compile=False)
                except Exception:
                    loaded_model = tf.keras.models.load_model(path)
                loaded[key] = loaded_model
                container.success(f"✅ Loaded after retry: {key}")
            except Exception as e2:
                err_msg2 = f"Could not load {key} model after retry: {e2}"
                st.error(f"🚫 {err_msg2}")
                log_msg(err_msg2)
                loaded[key] = None

    # Critical: main model is required to route to organ; if missing we cannot proceed
    if not loaded.get("main"):
        st.error("Critical error: 'main' model failed to load. App cannot continue.")
        log_msg("Critical: main model missing, stopping app.")
        # stop the app execution gracefully
        st.stop()
        raise RuntimeError("Main model failed to load.")

    st.success("🎉 Models load step finished (available models cached).")
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
    model = models.get("main")
    if model is None:
        raise RuntimeError("Main model not available.")
    preds = model.predict(img_tensor)
    idx = int(np.argmax(preds))
    return MAIN_CLASSES[idx], float(preds[0][idx])

def predict_domain(organ, img_tensor):
    model_domain = models.get(organ)
    classes = {
        "brain": BRAIN_CLASSES,
        "bone": BONE_CLASSES,
        "breast": BREAST_CLASSES,
        "kidney": KIDNEY_CLASSES
    }[organ]

    if model_domain is None:
        # graceful fallback: return "unknown" and zero confidence
        log_msg(f"predict_domain: model for '{organ}' is None. Using fallback.")
        return "unknown", 0.0

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
    try:
        img_buf = io.BytesIO()
        image.convert("RGB").save(img_buf, format="PNG")
        img_buf.seek(0)
        img_reader = ImageReader(img_buf)
        c.drawImage(img_reader, 25 * mm, height - 120 * mm, width=60 * mm, preserveAspectRatio=True)
    except Exception as e:
        log_msg(f"PDF image insert failed: {e}")

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
            # Predict main organ
            try:
                with st.spinner("Analyzing image (main model)..."):
                    tensor = preprocess_image(pil_img)
                    organ, conf_org = predict_main(tensor)
            except Exception as e:
                st.error(f"Error during main prediction: {e}")
                log_msg(f"Main prediction error: {e}")
                organ, conf_org = "unknown", 0.0

            # Predict domain (if model exists)
            finding, conf_find = predict_domain(organ, tensor) if organ != "unknown" else ("unknown", 0.0)

            st.success(f"Organ: {organ.upper()} ({conf_org*100:.1f}%) | Finding: {finding.upper()} ({conf_find*100:.1f}%)")

            # Generate LLM-based report or fallback
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
                        log_msg(f"Gemini error: {e}")
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
