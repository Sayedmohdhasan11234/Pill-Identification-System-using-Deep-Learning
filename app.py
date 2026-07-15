import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

# --- Page Configurations ---
st.set_page_config(
    page_title="Pill Identification System — Smart Medication Identifier",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Dataset Class Names ---
CLASS_NAMES = [
    'Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 
    'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep'
]

# Medical Information matching your 10 classes
DRUG_INFO = {
    'Alaxan': {
        "use": "Relief of mild to moderately severe pain, including muscle aches, backaches, and arthritis.",
        "type": "Analgesic / Anti-inflammatory",
        "schedule": "As needed every 4-6 hours"
    },
    'Bactidol': {
        "use": "Symptomatic relief of mouth and throat infections including pharyngitis and sore throat.",
        "type": "Antiseptic Oral Rinse",
        "schedule": "Gargle twice daily"
    },
    'Bioflu': {
        "use": "Relief of clogged nose, runny nose, cough, headache, and fever associated with common colds.",
        "type": "Antihistamine / Decongestant / Analgesic",
        "schedule": "Every 6 hours as needed"
    },
    'Biogesic': {
        "use": "Contains Paracetamol. Used for standard relief of minor aches, pains, and reducing fever.",
        "type": "Antipyretic / Analgesic",
        "schedule": "Every 4 to 6 hours"
    },
    'DayZinc': {
        "use": "Vitamin supplement containing Zinc and Vitamin C used to boost immune system health.",
        "type": "Nutritional Supplement",
        "schedule": "Once daily or as prescribed"
    },
    'Decolgen': {
        "use": "Commonly used for cold relief, congestion, runny nose, and headache.",
        "type": "Decongestant / Analgesic",
        "schedule": "3-4 times daily"
    },
    'Fish Oil': {
        "use": "Dietary supplement containing Omega-3 fatty acids for cardiovascular and joint support.",
        "type": "Dietary Supplement",
        "schedule": "Take with meals daily"
    },
    'Kremil S': {
        "use": "Antacid indicated for the relief of stomach hyperacidity, heartburn, and acid indigestion.",
        "type": "Antacid / Antiflatulent",
        "schedule": "1 hour after meals or during onset"
    },
    'Medicol': {
        "use": "Contains Ibuprofen. Target relief for structural pain, severe headaches, and inflammation.",
        "type": "NSAID (Pain Reliever)",
        "schedule": "Take with food every 4-6 hours"
    },
    'Neozep': {
        "use": "Indicated for the complete relief of cold symptoms, nasal congestion, and associated sinus pain.",
        "type": "Antihistamine / Decongestant",
        "schedule": "Every 6 hours"
    }
}

# --- Load Model with Caching ---
@st.cache_resource
def load_pill_model():
    try:
        with open("model (2).pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        return None

model = load_pill_model()

# --- Advanced Custom UI Styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; }
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
    }
    .hero-title { font-size: 2.8rem; font-weight: 700; margin-bottom: 5px; font-family: 'Inter', sans-serif; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.9; }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .sidebar-header { color: #1E3A8A; font-weight: 700; font-size: 1.2rem; }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown("<div class='sidebar-header'>🌐 System Status</div>", unsafe_allow_html=True)
    if model is not None:
        st.success("Core Network Operational")
    else:
        st.error("Model Not Found")
        
    st.write("---")
    st.markdown("### 🏷️ Detectable Database Matrix")
    st.caption("The computer vision layout is trained to extract features from the following classifications:")
    for name in CLASS_NAMES:
        st.markdown(f"• **{name}**")
    st.write("---")
    st.caption("Pill Identification System Core Framework v2.4 (EfficientNetB0 Engine)")

# --- Main App Header Banner ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">💊 Pill Identification System</div>
        <div class="hero-subtitle">Intelligent Computer Vision Auxiliary for Pharmaceutical Verification</div>
    </div>
    """, 
    unsafe_allow_html=True
)

if model is None:
    st.error("⚠️ **System Initialization Failure:** The pre-trained weights file `model (2).pkl` could not be loaded. Please ensure it is present in the server execution path.")
else:
    # Panels Setup
    left_panel, right_panel = st.columns([1, 1], gap="large")
    
    with left_panel:
        st.markdown("<h3 style='color:#1E3A8A;'>📸 Capture or Upload Target</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop pill image snapshot here...", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Specimen Input", use_container_width=True)
        else:
            st.info("💡 Pro-Tip: Ensure clean lighting and a clear overhead viewpoint of the individual capsule for maximum accuracy metrics.")

    with right_panel:
        st.markdown("<h3 style='color:#1E3A8A;'>📊 Neural Diagnostics</h3>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner("Decoding multidimensional visual maps..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                
                if len(img_array.shape) == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                
                img_batch = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_batch)
                
                if len(predictions.shape) > 1:
                    pred_array = predictions[0]
                else:
                    pred_array = predictions
                
                predicted_class_idx = np.argmax(pred_array)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                
                score = tf.nn.softmax(pred_array).numpy()
                confidence = float(np.max(score)) * 100

            # Confidence Alert Level Banners
            if confidence >= 85:
                st.markdown("<div style='background-color:#DCFCE7; border-left:5px solid #22C55E; padding:15px; border-radius:8px; color:#14532D; font-weight:bold;'>High Certainty Verification Match Found</div>", unsafe_allow_html=True)
            elif confidence >= 60:
                st.markdown("<div style='background-color:#FEF9C3; border-left:5px solid #EAB308; padding:15px; border-radius:8px; color:#713F12; font-weight:bold;'>Moderate Level Confidence Match</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#FEE2E2; border-left:5px solid #EF4444; padding:15px; border-radius:8px; color:#7F1D1D; font-weight:bold;'>Low Certainty Verification Warning</div>", unsafe_allow_html=True)

            # Elegant Result Information Display
            st.markdown(
                f"""
                <div class="card">
                    <span style="color:#64748B; font-size:0.9rem; text-transform:uppercase; font-weight:600;">Identified Label Classification</span>
                    <h2 style="color:#1E3A8A; margin:0 0 10px 0;">{predicted_class}</h2>
                    <span style="color:#64748B; font-size:0.9rem; font-weight:600;">Match Structural Integrity Score</span>
                    <h3 style="color:#3B82F6; margin:0 0 5px 0;">{confidence:.2f}%</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.progress(int(confidence))
            
            # Interactive Information Breakdowns
            tab1, tab2, tab3 = st.tabs(["📋 Clinical Profile", "⏰ Standard Routine", "🛑 Medical Disclaimer"])
            
            with tab1:
                info = DRUG_INFO.get(predicted_class, {})
                st.markdown(f"**Classification Family:** `{info.get('type', 'Unknown')}`")
                st.write(f"**Primary Medical Indication:** {info.get('use', 'No clinical usage documentation recorded.')}")
                
            with tab2:
                schedule_time = DRUG_INFO.get(predicted_class, {}).get('schedule', 'Consult medical specialist.')
                st.write(f"**Suggested Administration Interval:** {schedule_time}")
                
            with tab3:
                st.caption("⚠️ **Machine Learning Notice:** This verification module acts exclusively as an secondary educational assistant. Never change medication doses or schedules based solely on computer vision model suggestions. Consult your physician or local pharmacist.")
        else:
            st.markdown(
                """
                <div style="background-color: white; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 40px; text-align: center; color: #64748B;">
                    📥 Awaiting Target input sequence to analyze metrics.
                </div>
                """, 
                unsafe_allow_html=True
            )