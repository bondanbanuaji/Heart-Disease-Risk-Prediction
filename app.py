import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- Page Configuration ---
st.set_page_config(
    page_title="HeartRisk AI | Dark Mode",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Config ---
MODEL_PATH = 'C45_pinjam_mod.pkl'

# Fixed mappings based on DecisionTree_231351030.ipynb LabelEncoder.classes_
LABEL_ENCODINGS = {
    'Age': {'Senior': 1, 'Middle-aged': 0, 'Young': 2},
    'Cholesterol Level': {'High': 0, 'Low': 1, 'Normal': 2},
    'Blood Pressure': {'High': 0, 'Low': 1, 'Normal': 2},
    'Smoking': {'Yes': 1, 'No': 0},
    'Physical Activity': {'High': 0, 'Low': 1, 'Moderate': 2},
    'BMI': {'Normal': 0, 'Obese': 1, 'Overweight': 2},
}

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}")
        return None

# --- UI Helper Functions ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* Base Dark Theme (Slate 950) */
        .stApp {
            background-color: #020617;
            color: #f8fafc;
        }
        
        /* Typography */
        h1, h2, h3, h4, p, label, .stMarkdown {
            color: #f8fafc !important;
        }

        .main-header {
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            padding: 2.5rem 0;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            letter-spacing: -0.05em;
        }
        
        .sub-header {
            text-align: center;
            color: #94a3b8 !important;
            font-size: 1.25rem;
            margin-bottom: 3.5rem;
            font-weight: 400;
        }

        /* Glassmorphism Cards - High Contrast for Dark Mode */
        .glass-card {
            background: rgba(30, 41, 59, 0.5); /* Slate 800 with opacity */
            backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
            margin-bottom: 24px;
        }

        /* Dynamic Prediction Result Cards */
        .prediction-card {
            padding: 48px;
            border-radius: 32px;
            color: white !important;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin: 24px 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .prediction-card:hover { 
            transform: scale(1.02); 
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
        }
        .prediction-card h1 { color: white !important; }
        
        .risk-high { background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%); }
        .risk-medium { background: linear-gradient(135deg, #78350f 0%, #f59e0b 100%); }
        .risk-low { background: linear-gradient(135deg, #064e3b 0%, #10b981 100%); }
        
        /* Selectbox & Input Overrides for Dark Visibility */
        .stSelectbox label {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 8px;
            color: #cbd5e1 !important;
        }
        
        /* Premium Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            color: white;
            padding: 18px 36px;
            border-radius: 16px;
            font-weight: 700;
            font-size: 1.1rem;
            width: 100%;
            margin-top: 1rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #0f172a;
        }

        /* Chart Tooltips Fix */
        .plotly-notifier {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

def display_dashboard(model):
    st.markdown("### 🧬 Biological & Lifestyle Markers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        age = st.selectbox('👤 Age Category', list(LABEL_ENCODINGS['Age'].keys()))
        cholesterol = st.selectbox('💉 HDL/LDL Ratio', list(LABEL_ENCODINGS['Cholesterol Level'].keys()))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        bp = st.selectbox('🩸 Blood Pressure (Sys/Dia)', list(LABEL_ENCODINGS['Blood Pressure'].keys()))
        smoking = st.selectbox('🚬 Tobacco Usage', list(LABEL_ENCODINGS['Smoking'].keys()))
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        activity = st.selectbox('🏃 Daily Activity Level', list(LABEL_ENCODINGS['Physical Activity'].keys()))
        bmi = st.selectbox('⚖️ Body Mass Index (BMI)', list(LABEL_ENCODINGS['BMI'].keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    input_data = np.array([
        LABEL_ENCODINGS['Age'][age],
        LABEL_ENCODINGS['Cholesterol Level'][cholesterol],
        LABEL_ENCODINGS['Blood Pressure'][bp],
        LABEL_ENCODINGS['Smoking'][smoking],
        LABEL_ENCODINGS['Physical Activity'][activity],
        LABEL_ENCODINGS['BMI'][bmi],
    ]).reshape(1, -1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 INITIALIZE AI ANALYSIS"):
        with st.spinner('Accessing Neural Model...'):
            prediction = model.predict(input_data)[0]
            risk_mapping = {0: 'Critical - High Risk', 2: 'Elevated - Medium Risk', 1: 'Safe - Low Risk'}
            risk_classes = {0: 'risk-high', 2: 'risk-medium', 1: 'risk-low'}
            st.markdown(f'<div class="prediction-card {risk_classes[prediction]}"><p style="margin:0; text-transform:uppercase; font-weight:700; letter-spacing:0.1em; opacity:0.8;">Detection Status</p><h1>{risk_mapping[prediction]}</h1></div>', unsafe_allow_html=True)

    return {'Age': age, 'Cholesterol': cholesterol, 'BP': bp, 'Smoking': smoking, 'Activity': activity, 'BMI': bmi}

def display_analysis(data):
    st.markdown("### 📊 Parameter Intensity Vector")
    risk_scores = {'High': 1.0, 'Obese': 1.0, 'Low': 1.0, 'Yes': 1.0, 'Senior': 1.0, 'Moderate': 0.5, 'Overweight': 0.7, 'Middle-aged': 0.5, 'Normal': 0.2, 'Young': 0.1, 'No': 0.0}
    
    df = pd.DataFrame({
        'Parameter': list(data.keys()), 
        'Risk Intensity': [risk_scores.get(v, 0.5) for v in data.values()]
    })
    
    # Dark Mode Bar Chart
    fig = px.bar(df, x='Parameter', y='Risk Intensity', color='Risk Intensity', 
                 color_continuous_scale='RdYlGn_r', range_y=[0, 1.1],
                 template='plotly_dark')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dark Mode Radar Chart
    fig_polar = go.Figure(data=go.Scatterpolar(
        r=df['Risk Intensity'].tolist() + [df['Risk Intensity'].iloc[0]],
        theta=df['Parameter'].tolist() + [df['Parameter'].iloc[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#60a5fa', width=3)
    ))
    fig_polar.update_layout(
        template='plotly_dark',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#334155')
        ),
        showlegend=False, 
        paper_bgcolor='rgba(0,0,0,0)',
        title="Cardio-Risk Fingerprint"
    )
    st.plotly_chart(fig_polar, use_container_width=True)

def display_insights(model):
    st.markdown("### 🧠 Symbolic AI Transparency")
    st.write("Understand how the **C4.5 Optimized Decision Tree** processed your specific telemetry data.")
    
    if st.checkbox("Reveal Branch Logic Architecture"):
        # Custom Matplotlib style for Dark Mode
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
        plot_tree(model, 
                 feature_names=list(LABEL_ENCODINGS.keys()), 
                 class_names=['High Risk', 'Low Risk', 'Medium Risk'], 
                 filled=True, rounded=True, 
                 fontsize=11, proportion=True, precision=2)
        
        # Make the plot frame transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        st.pyplot(fig)

def main():
    inject_custom_css()
    st.markdown('<h1 class="main-header">HeartRisk AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Premium Machine Learning Protocol for Cardiovascular Diagnostics</p>', unsafe_allow_html=True)
    
    model = load_model()
    if model is None: return
    
    tab1, tab2, tab3 = st.tabs(["⚡ Diagnostic Portal", "📡 Telemetry Data", "⚙️ Core Protocol"])
    with tab1: user_data = display_dashboard(model)
    with tab2: display_analysis(user_data)
    with tab3: display_insights(model)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div style="text-align: center; color: #475569; font-size: 0.85rem; letter-spacing: 0.1em;">© 2026 HEARTWISE CLINICAL SYSTEMS | SECURE PROTOCOL</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()