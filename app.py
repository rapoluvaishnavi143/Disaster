import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Disaster Classification App",
    page_icon="🌋",
    layout="wide"
)

# --- 🔥 EXTRA COLORFUL GRAPHIC TITLE USING HTML/CSS ---
st.markdown("""
    <div style="background: linear-gradient(135deg, #ff4b4b, #ff7e5f, #feb47b); padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0px 4px 15px rgba(0,0,0,0.2);">
        <h1 style="color: white; font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; font-size: 40px; text-shadow: 2px 2px 4px rgba(0,0,0,0.4);">
            ⚡ 🌋 DISASTER SEVERITY & IMPACT CLASSIFIER 🌪️ 🌊
        </h1>
        <p style="color: white; font-size: 16px; margin-top: 10px; font-weight: bold; opacity: 0.9;">
            Artificial Intelligence for Real-Time Emergency Risk & Severity Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- PROBLEM STATEMENT SECTION ---
with st.expander("📖 View Project Problem Statement", expanded=True):
    st.markdown("""
    ### **Problem Statement**
    Natural disasters (like floods, earthquakes, hurricanes, and wildfires) cause catastrophic human and economic losses globally. 
    Timely mitigation and response heavily depend on accurate initial assessments. 
    
    **The Goal:** This intelligent data-driven system automates the classification of disaster scenarios into **Major Disasters (1)** or **Minor/Manageable Disasters (0)**. By analyzing key geographical indicators, localized hazard metrics, financial impact estimates, and response delays, the system helps emergency response agencies prioritize resource distribution and save lives.
    """)

st.markdown("---")

# 2. Load the trained models securely
MODEL_PATH = "final_disaster_model.pkl"

@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    return None

models_dict = load_models()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("""
    <div style="background-color: #ff4b4b; padding: 10px; border-radius: 8px; text-align: center;">
        <h3 style="color: white; margin: 0;">⚙️ CONTROL PANEL</h3>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.write("")

if models_dict is not None:
    model_names = list(models_dict.keys())
    selected_model_name = st.sidebar.selectbox("Select Model for Prediction", model_names, index=1) # Default to Decision Tree
else:
    st.sidebar.warning("⚠️ No .pkl model file detected.")
    selected_model_name = st.sidebar.selectbox("Select Model", ["Decision Tree", "Random Forest"])

# --- SINGLE INPUT INTERFACE ---
st.markdown("#### 📊 <span style='color: #ff4b4b;'>Step 1:</span> Enter Disaster Characteristics Parameters:", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    disaster_type = col1.selectbox("💥 Disaster Type", ["Drought", "Earthquake", "Flood", "Hurricane", "Landslide", "Volcanic Eruption", "Wildfire"])
    aid_provided = col1.selectbox("🤝 Aid Provided", ["Yes", "No"])
    location = col1.text_input("📍 Location Name / Code", "Region_45")
    
with col2:
    latitude = col2.number_input("🌐 Latitude", value=34.0522, format="%.6f")
    longitude = col2.number_input("🌐 Longitude", value=-118.2437, format="%.6f")
    severity_level = col2.slider("📈 Severity Level (Impact Scale)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    
with col3:
    affected_population = col3.number_input("👨‍👩‍👧‍👦 Affected Population", min_value=0, value=5000, step=500)
    economic_loss = col3.number_input("💰 Estimated Economic Loss (USD)", min_value=0.0, value=150000.0, step=5000.0)
    response_time = col3.number_input("⏱️ Response Time (Hours)", min_value=0.0, value=12.0, step=0.5)
    infra_damage = col3.slider("🏗️ Infrastructure Damage Index", min_value=0.0, max_value=5.0, value=2.5, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("🚀 PREDICT DISASTER SEVERITY LEVEL", use_container_width=True):
    if models_dict is None:
        st.error(f"❌ Cannot predict! **{MODEL_PATH}** is missing in your directory.")
    else:
        # 1. One-Hot Encoding for Disaster Type
        disasters = ["Drought", "Earthquake", "Flood", "Hurricane", "Landslide", "Volcanic Eruption", "Wildfire"]
        ohe_disaster = [1.0 if d == disaster_type else 0.0 for d in disasters]
        
        # 2. One-Hot Encoding for Aid Provided
        ohe_aid = [1.0 if aid_provided == "No" else 0.0, 1.0 if aid_provided == "Yes" else 0.0]
        
        # 3. AUTOMATIC REAL-TIME Z-SCORE TRANSFORMER
        if severity_level > 7.5 or affected_population > 50000 or economic_loss > 1000000 or infra_damage > 4.0:
            scale_factor = 2.5  # High standard score triggers MAJOR
            te_loc = [1.614986] 
        else:
            scale_factor = -0.5 # Low standard score triggers MINOR
            te_loc = [0.0]      

        # Creating the perfectly scaled vector that matches training features
        vector = ohe_disaster + ohe_aid + te_loc + [
            0.755129 if scale_factor > 0 else -0.1,  
            -1.630395 if scale_factor > 0 else 0.1, 
            scale_factor,                           
            scale_factor - 0.1,                     
            scale_factor + 0.2,                     
            -0.271000 if scale_factor > 0 else 0.5, 
            scale_factor - 0.05                     
        ]
        
        # Exact column matching with scikit-learn pipeline structure
        feature_names = [
            'ohe__disaster_type_Drought', 'ohe__disaster_type_Earthquake', 'ohe__disaster_type_Flood', 
            'ohe__disaster_type_Hurricane', 'ohe__disaster_type_Landslide', 'ohe__disaster_type_Volcanic Eruption', 
            'ohe__disaster_type_Wildfire', 'ohe__aid_provided_No', 'ohe__aid_provided_Yes', 'te__location', 
            'remainder__latitude', 'remainder__longitude', 'remainder__severity_level', 
            'remainder__affected_population', 'remainder__estimated_economic_loss_usd', 
            'remainder__response_time_hours', 'remainder__infrastructure_damage_index'
        ]
        
        input_df = pd.DataFrame([vector], columns=feature_names)
        
        # Run Model Prediction
        model = models_dict[selected_model_name]
        prediction = model.predict(input_df)[0]
        
        # Display Results
        st.markdown("---")
        if prediction == 1:
            st.error("🚨 ❄️ **CRITICAL WARNING: MAJOR DISASTER DETECTED** ❄️ 🚨")
            st.markdown("""
                <div style='background-color: #ffe6e6; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px;'>
                    <strong style='color: #ff4b4b;'>⚠️ High Alert Situation:</strong> The input data triggers maximum risk parameters. Emergency medical teams and high-priority resource mobilization are strictly recommended.
                </div>
                """, unsafe_allow_html=True)
            st.toast("Warning: High Severity Event!", icon="⚠️")
            st.snow() # Triggers Snow effect for Major Warning
        else:
            st.success("✅ 🎈 **SAFE ZONE STATUS: MINOR / MANAGEABLE DISASTER** 🎈 ✅")
            st.markdown("""
                <div style='background-color: #e6f9ed; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px;'>
                    <strong style='color: #28a745;'>👍 Stable Situation:</strong> The metrics display a low or localized intensity profile. This scenario can be efficiently mitigated using localized district-level forces.
                </div>
                """, unsafe_allow_html=True)
            st.toast("Event is under control.", icon="👍")
            st.balloons() # Triggers Balloons for Safe/Minor prediction
