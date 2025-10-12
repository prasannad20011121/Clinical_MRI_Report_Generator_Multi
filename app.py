# app.py
import os
import io
import re
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
import requests # Import requests for downloading

# ======================
# Gemini Integration
# ======================
USE_GEMINI = True
try:
    import google.generativeai as genai
except ImportError:
    USE_GEMINI = False

# ======================
# Streamlit Config
# ======================
st.set_page_config(page_title="AI Medical Report Generator", layout="wide")
st.title("AI Medical Report Generator")
st.caption("Upload a medical image → AI Diagnosis → LLM-based report → Download professional PDF")

# ======================
# Model Setup
# ======================
MODEL_DIR = "models"
IMG_SIZE = (224, 224)

MAIN_CLASSES = ['bone', 'brain', 'breast', 'kidney']
BRAIN_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BONE_CLASSES = ['fractured', 'not fractured']
BREAST_CLASSES = ['benign', 'malignant']
KIDNEY_CLASSES = ['cyst', 'normal', 'stone', 'tumor']

# --- NEW: Google Drive Model URLs ---
# Note: The URLs have been changed from /view to /uc?id= for direct download
MODEL_URLS = {
    "main": "https://drive.google.com/uc?id=1MrmfGNWW6Msz71WTcrCJcouk5vyDWhMq",
    "brain": "https://drive.google.com/uc?id=1MFRWHTsp830qpVFm19x-74gQ3h6XsJ73",
    "bone": "https://drive.google.com/uc?id=1cFVYwUz8rVqu6gjlMW-_wYoukyCpto5h",
    "breast": "https://drive.google.com/uc?id=1aQ327zLaqHqKrw30qOXlOYPPW3NScFDU",
    "kidney": "https://drive.google.com/uc?id=1ZAmC8nssodO5IVWhUMpmpOxxdoIg1H2d"
}

MODEL_PATHS = {
    "main": os.path.join(MODEL_DIR, "main_model.keras"),
    "brain": os.path.join(MODEL_DIR, "brain_model.keras"),
    "bone": os.path.join(MODEL_DIR, "bone_model.keras"),
    "breast": os.path.join(MODEL_DIR, "breast_model.keras"),
    "kidney": os.path.join(MODEL_DIR, "kidney_model.keras")
}

# --- NEW: Function to download files from Google Drive ---
def download_file_from_google_drive(url, destination):
    """Downloads a file from a Google Drive URL to a destination."""
    with st.spinner(f"Downloading model: {os.path.basename(destination)}..."):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Downloaded {os.path.basename(destination)} successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {os.path.basename(destination)}: {e}")
            raise

# ======================
# Load Models
# ======================
@st.cache_resource
def load_all_models():
    """Downloads models if not present, then loads all required models."""
    st.info("Loading AI models...")
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    models = {}
    try:
        for key, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                st.warning(f"Model for '{key}' not found locally. Downloading from Google Drive...")
                download_file_from_google_drive(MODEL_URLS[key], path)
            models[key] = tf.keras.models.load_model(path)
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"Fatal Error during model loading: {e}. Please check the model files and paths.")
        st.stop()
    return models

models = load_all_models()

# ======================
# Gemini Setup
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if USE_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-1.5-pro-latest") # Updated model name
else:
    llm_model = None
    st.warning("⚠️ Gemini API key not found. You can set it in the Streamlit Cloud secrets. The app will use a basic report format.")

# ======================
# Helper Functions (NO CHANGES BELOW THIS LINE until the prompt)
# ======================
def preprocess_image(pil_img):
    """Prepares the uploaded image for model prediction."""
    img = pil_img.convert("L").resize(IMG_SIZE) # Convert to grayscale and resize
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1)) # Add batch and channel dimensions
    return arr

def predict_main(img_tensor):
    """Predicts the main organ category."""
    preds = models["main"].predict(img_tensor)
    idx = int(np.argmax(preds))
    return MAIN_CLASSES[idx], float(preds[0][idx])

def predict_domain(organ, img_tensor):
    """Predicts the specific finding for a given organ."""
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

def local_report(organ, finding, mode):
    """Generates a more detailed, structured fallback report when the LLM is unavailable."""
    today = datetime.datetime.now().strftime("%d-%b-%Y")
    report_id = str(uuid.uuid4())[:8].upper()

    if mode == "Doctor Mode":
        text = f"""
**PATIENT:** [Patient Name]
**MRN:** [Medical Record Number]
**DATE OF SERVICE:** {today}

**EXAMINATION:**
AI-Assisted Radiographic Analysis of the {organ.capitalize()}.

**FINDINGS:**
The automated analysis of the provided imaging data reveals characteristics highly suggestive of a "{finding.lower()}" within the {organ}. The features noted by the model include [e.g., abnormal signal intensity, a clear discontinuity of the cortical margin, a well-defined mass with specific border characteristics]. These findings are localized to the [e.g., distal metaphysis, specific lobe or quadrant]. No other acute abnormalities were flagged by the system.

**IMPRESSION:**
The preliminary AI finding is a {finding.capitalize()}. This represents a significant observation that requires immediate clinical attention and further diagnostic workup.

**CLINICAL CORRELATION:**
It is imperative to correlate these AI-driven findings with the patient's clinical presentation, history, and physical examination. Laboratory results and prior imaging studies should also be reviewed to provide context to this automated analysis.

**RECOMMENDATIONS:**
1.  **Immediate Specialist Consultation:** An urgent referral to a specialist (e.g., Orthopedist, Neurologist, Oncologist) is strongly recommended for definitive evaluation.
2.  **Confirmatory Imaging:** Consider advanced imaging modalities (e.g., CT, MRI, Ultrasound) to better characterize the finding and guide potential intervention.
3.  **Biopsy if Indicated:** A biopsy may be necessary for histopathological confirmation, depending on the clinical scenario.
4.  **Monitoring:** Close clinical and radiographic follow-up is advised.
"""
    else:  # Patient Mode
        text = f"""
**SUMMARY OF YOUR AI-ASSISTED SCAN**
**Date:** {today}
**Scan Type:** {organ.capitalize()} Scan Analysis
**AI Detected Finding:** {finding.capitalize()}

**WHAT THE AI FOUND:**
Our AI system carefully analyzed your medical scan and identified patterns that suggest the presence of a "{finding.lower()}" in the {organ} area. The system highlights this as an area that needs further attention from your medical team.

**WHAT THIS MEANS FOR YOU:**
This is a preliminary finding, not a final diagnosis. Think of this AI result as an advanced tool that helps your doctor focus on a specific area of interest. It provides valuable information that, when combined with your doctor's expertise, helps build a complete picture of your health. The next step is a thorough review by your healthcare provider to confirm and understand these results in the context of your overall health.

**RECOMMENDED NEXT STEPS:**
Your doctor will discuss these findings with you in detail. They may suggest further tests, such as more advanced scans or other procedures, to get more information. It is essential to follow their guidance for the most accurate diagnosis and treatment plan.

**DO'S AND DON'TS:**
**Do:**
-  Schedule a follow-up appointment with your doctor to discuss this report in detail.
-  Prepare a list of any questions or concerns you may have for your doctor.
-  Continue to follow any current medical advice or treatment plans unless instructed otherwise by your provider.

**Don't:**
-  Do not interpret this report as a final diagnosis or a reason to panic. It is a tool to guide your doctor.
-  Do not start, stop, or change any medications or treatments based solely on this AI report.
-  Avoid searching for information online that may cause unnecessary anxiety; rely on your doctor for accurate information.
"""
    return text.strip()

def generate_pdf(report_text, image, organ, organ_conf=None, finding_conf=None):
    """Generates a professional, well-formatted PDF report from text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Header ---
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 40*mm, "AI MEDICAL DIAGNOSTIC REPORT")
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, height - 45*mm, "Generated by CNN + Gemini LLM System")

    # --- Report Metadata ---
    c.setFont("Helvetica", 10)
    today = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    report_id = str(uuid.uuid4())[:8].upper()
    c.drawString(25*mm, height - 60*mm, f"Date: {today}")
    c.drawRightString(width - 25*mm, height - 60*mm, f"Report ID: {report_id}")
    c.line(25*mm, height - 65*mm, width - 25*mm, height - 65*mm)

    # --- Image and AI Summary ---
    img_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="PNG")
    img_buf.seek(0)
    img_reader = ImageReader(img_buf)
    c.drawImage(img_reader, 25*mm, height - 125*mm, width=60*mm, preserveAspectRatio=True)

    summary_x = 100 * mm
    summary_y = height - 85 * mm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(summary_x, summary_y, "AI Prediction Summary")
    c.setFont("Helvetica", 11)
    summary_y -= 18
    c.drawString(summary_x, summary_y, f"Organ Analyzed: {organ.capitalize()}")
    summary_y -= 14
    if organ_conf:
        c.drawString(summary_x, summary_y, f"Confidence (Organ): {organ_conf*100:.2f}%")
    summary_y -= 14
    if finding_conf:
        c.drawString(summary_x, summary_y, f"Confidence (Finding): {finding_conf*100:.2f}%")
        
    c.line(25*mm, height - 140*mm, width - 25*mm, height - 140*mm)

    # --- Detailed Report Text (with intelligent formatting) ---
    margin_left = 25 * mm
    margin_right = 25 * mm
    y_cursor = height - 150 * mm
    available_width = width - margin_left - margin_right
    line_height_normal = 14
    line_height_heading = 18

    paragraphs = report_text.split('\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            y_cursor -= line_height_normal * 0.5 # Add space for empty lines
            continue

        is_heading = para.startswith('**') and para.endswith('**')
        
        if is_heading:
            c.setFont("Helvetica-Bold", 12)
            text = para.strip('* ')
            y_cursor -= line_height_heading
            c.drawString(margin_left, y_cursor, text)
            y_cursor -= 5 # Extra space after heading
            c.setFont("Helvetica", 11)
        else:
            words = para.split()
            line = ''
            for word in words:
                if c.stringWidth(line + ' ' + word, "Helvetica", 11) <= available_width:
                    line += ' ' + word
                else:
                    y_cursor -= line_height_normal
                    c.drawString(margin_left, y_cursor, line.strip())
                    line = word
            y_cursor -= line_height_normal
            c.drawString(margin_left, y_cursor, line.strip())
            y_cursor -= line_height_normal * 0.5 # Space after paragraph

        if y_cursor < 40 * mm:
            c.showPage()
            y_cursor = height - 40 * mm
            c.setFont("Helvetica", 11)

    # --- Footer ---
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2, 25*mm, "This AI report is for research use only. Always confirm results with a medical professional.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ======================
# Streamlit UI
# ======================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Upload Medical Image")
    uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
    mode = st.radio("Select Report Mode", ["Doctor Mode", "Patient Mode"], horizontal=True)

with col2:
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Report", type="primary"):
            with st.spinner("Analyzing image..."):
                tensor = preprocess_image(pil_img)
                organ, conf_org = predict_main(tensor)
                finding, conf_find = predict_domain(organ, tensor)
                st.success(f"Organ: {organ.upper()} ({conf_org*100:.1f}%) | Finding: {finding.upper()} ({conf_find*100:.1f}%)")

            with st.spinner("Generating detailed report with Gemini AI..."):
                report_text = ""
                if llm_model:
                    try:
                        # --- PROMPT IS THE SAME ---
                        prompt = f"""
                        Act as a senior diagnostic radiologist AI. Your task is to generate a highly detailed, comprehensive, and well-structured medical report for a '{mode}'.

                        **AI Model's Preliminary Findings:**
                        - Organ Analyzed: {organ}
                        - Suspected Condition: {finding}

                        **Instructions for Report Generation:**
                        1.  **Tone & Language:**
                            - For 'Doctor Mode': Use precise, formal medical terminology. Be thorough and professional.
                            - For 'Patient Mode': Use clear, simple, and empathetic language. Avoid jargon and explain complex terms.

                        2.  **Structure & Formatting:** Strictly use double asterisks for main headings (e.g., **FINDINGS**). Do not use any other markdown.

                        3.  **Content - Be Expansive and Detailed in All Sections:**
                            - **PATIENT/MRN:** Use placeholders like [Patient Name] and [Medical Record Number].
                            - **EXAMINATION:** State the procedure (e.g., "AI-Assisted Analysis of a Radiograph of the {organ.capitalize()}").
                            - **FINDINGS:** **Elaborate in significant detail.** Describe the typical radiographic appearance of '{finding}' in the '{organ}'. Discuss its characteristics, such as location, morphology, borders, and impact on surrounding structures. Be as descriptive as possible, as if you are describing the image to another physician.
                            - **IMPRESSION:** Provide a clear, concise diagnostic conclusion based on the findings.
                            - **RECOMMENDATIONS:** Provide a numbered list of specific, actionable next steps. Go beyond generic advice. Suggest specific further imaging, consultations, or tests.

                        4.  **Mode-Specific Sections (Include these):**
                            - **If 'Doctor Mode':**
                                - **CLINICAL CORRELATION:** Stress the importance of correlating AI findings with the patient's symptoms, history, and other clinical data.
                                - **DIFFERENTIAL DIAGNOSIS:** List and briefly explain other possible conditions that could present with similar findings. This is crucial for a professional report.
                            - **If 'Patient Mode':**
                                - **WHAT THIS MEANS FOR YOU:** A dedicated paragraph explaining the findings in simple terms and managing patient expectations.
                                - **DO'S AND DON'TS:** A clear, bulleted or numbered list of what the patient should and should not do.

                        5.  **Exclusions:** Do NOT include any disclaimers, warnings, or emojis in your response. The output should be only the professional report text itself.
                        """
                        response = llm_model.generate_content(prompt)
                        report_text = response.text.strip()
                    except Exception as e:
                        st.error(f"An error occurred with the LLM: {e}")
                        report_text = local_report(organ, finding, mode)
                
                if not report_text: # Fallback if LLM fails or is disabled
                    st.warning("LLM generation failed or is disabled. Using a template-based report.")
                    report_text = local_report(organ, finding, mode)

            st.subheader("Generated Report")
            st.text_area("Medical Report", value=report_text, height=450)

            pdf_data = generate_pdf(report_text, pil_img, organ, conf_org, conf_find)
            st.download_button(
                label="⬇️ Download Full Medical Report (PDF)",
                data=pdf_data,
                file_name=f"{organ}_report_{str(uuid.uuid4())[:4]}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Please upload a medical image to generate your AI-powered report.")

st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This AI system is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a certified doctor for any medical concerns.")