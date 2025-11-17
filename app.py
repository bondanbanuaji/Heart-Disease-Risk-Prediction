import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd

# Set page config for better UI
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    model_path = 'C45_pinjam_mod.pkl'
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

loaded_model = load_model()

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .prediction-card {
        background: linear-gradient(145deg, #667eea, #764ba2);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .risk-high { background: linear-gradient(145deg, #e74c3c, #c0392b); }
    .risk-medium { background: linear-gradient(145deg, #f39c12, #d35400); }
    .risk-low { background: linear-gradient(145deg, #27ae60, #2ecc71); }
    .feature-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #3498db;
    }
    .sidebar .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation effect
st.markdown('<h1 class="main-header">❤️ Advanced Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7f8c8d; margin-bottom: 30px;'>Using C4.5 Decision Tree Algorithm with Interactive 3D Visualizations</h3>", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📊 Analysis", "📈 Model Insights"])

with tab1:
    # Feature input section with enhanced UI
    st.markdown("### 🏥 Patient Information")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        age = st.selectbox('👤 Age Group', ('Senior', 'Middle-aged', 'Young'), help="Select your age group")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        cholesterol = st.selectbox('💉 Cholesterol Level', ('High', 'Normal', 'Low'), help="Select your cholesterol level")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        blood_pressure = st.selectbox('🩸 Blood Pressure', ('High', 'Normal', 'Low'), help="Select your blood pressure level")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        smoking = st.selectbox('🚬 Smoking Habit', ('Yes', 'No'), help="Do you smoke?")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        physical_activity = st.selectbox('🏃 Physical Activity', ('Low', 'Moderate', 'High'), help="Select your physical activity level")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        bmi = st.selectbox('⚖️ BMI Category', ('Overweight', 'Normal', 'Obese'), help="Select your BMI category")
        st.markdown('</div>', unsafe_allow_html=True)

    # Create a mapping for the labels to numbers (as per how the model was trained)
    label_encodings = {
        'Age': {'Senior': 1, 'Middle-aged': 0, 'Young': 2},
        'Cholesterol Level': {'High': 0, 'Normal': 1, 'Low': 2},
        'Blood Pressure': {'High': 0, 'Normal': 1, 'Low': 2},
        'Smoking': {'Yes': 1, 'No': 0},
        'Physical Activity': {'Low': 0, 'Moderate': 1, 'High': 2},
        'BMI': {'Overweight': 1, 'Normal': 0, 'Obese': 2},
    }

    # Encode the input data
    new_data = np.array([
        label_encodings['Age'][age],
        label_encodings['Cholesterol Level'][cholesterol],
        label_encodings['Blood Pressure'][blood_pressure],
        label_encodings['Smoking'][smoking],
        label_encodings['Physical Activity'][physical_activity],
        label_encodings['BMI'][bmi],
    ]).reshape(1, -1)

    # Risk assessment button with animation effect
    st.markdown("### 🎯 Risk Assessment")
    if st.button("🔮 Predict Risk Level", type="primary", use_container_width=True):
        with st.spinner('Analyzing your health data... 🧪'):
            prediction = loaded_model.predict(new_data)
            prediction_proba = loaded_model.predict_proba(new_data) if hasattr(loaded_model, 'predict_proba') else None

            risk_mapping = {0: 'High Risk', 1: 'Low Risk', 2: 'Medium Risk'}
            risk_colors = {0: 'risk-high', 1: 'risk-low', 2: 'risk-medium'}

            predicted_risk = risk_mapping[prediction[0]]
            risk_color = risk_colors[prediction[0]]

            # Display prediction with animated card
            st.markdown(f"""
            <div class="prediction-card {risk_color}">
                <h2>📊 Prediction Result</h2>
                <h1>❤️ {predicted_risk}</h1>
                <p style="font-size: 1.2em;">Based on your health parameters</p>
            </div>
            """, unsafe_allow_html=True)

            # Display probability if available
            if prediction_proba is not None:
                prob_score = prediction_proba[0][prediction[0]]
                st.progress(float(prob_score))
                st.markdown(f"**Confidence Score:** {prob_score:.2%}")

with tab2:
    st.markdown("### 📊 Health Parameter Analysis")

    # Create a DataFrame for visualization
    features = ['Age', 'Cholesterol', 'Blood Pressure', 'Smoking', 'Physical Activity', 'BMI']
    values = [age, cholesterol, blood_pressure, smoking, physical_activity, bmi]

    df = pd.DataFrame({
        'Feature': features,
        'Value': values,
        'Risk_Factor': [1 if v in ['High', 'Yes', 'Low', 'Overweight', 'Obese'] else 0.5 if v in ['Normal'] else 0
                       for v in values]  # Simplified risk factor calculation
    })

    # Create interactive Plotly chart
    fig = px.bar(df, x='Feature', y='Risk_Factor', color='Value',
                 title='Health Parameter Risk Assessment',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Create radar chart for health parameters
    fig2 = go.Figure()

    # Normalize values for radar chart
    norm_values = []
    for val in values:
        if val in ['High', 'Yes', 'Low', 'Overweight', 'Obese']:
            norm_values.append(1)
        elif val in ['Normal']:
            norm_values.append(0.5)
        else:
            norm_values.append(0)

    fig2.add_trace(go.Scatterpolar(
        r=norm_values + [norm_values[0]],  # Close the loop
        theta=features + [features[0]],
        fill='toself',
        name='Health Parameters'
    ))

    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Health Parameters Radar Chart"
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### 🌳 Decision Tree Visualization")

    # Create an interactive decision tree visualization
    st.markdown("""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h4>🔍 Model Insights</h4>
        <p>The C4.5 Decision Tree algorithm analyzes your health parameters to predict heart disease risk.</p>
        <ul>
            <li><strong>Age:</strong> Senior patients have higher risk factors</li>
            <li><strong>Cholesterol:</strong> High levels increase risk significantly</li>
            <li><strong>Blood Pressure:</strong> High BP is a major risk factor</li>
            <li><strong>Smoking:</strong> Increases heart disease risk substantially</li>
            <li><strong>Physical Activity:</strong> Regular exercise reduces risk</li>
            <li><strong>BMI:</strong> Overweight/obese conditions increase risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Display the decision tree
    if st.button("Show Decision Tree Visualization"):
        fig = plt.figure(figsize=(15, 10))
        plot_tree(loaded_model,
            feature_names=[
                'Age',
                'Cholesterol Level',
                'Blood Pressure',
                'Smoking',
                'Physical Activity',
                'BMI',
            ],
            class_names=[
                'High Risk',
                'Low Risk',
                'Medium Risk',
            ],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Heart Disease Risk Decision Tree", fontsize=16, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("💡 **Tip:** Regular exercise can reduce heart disease risk by up to 35%")
with col2:
    st.warning("⚠️ **Note:** This prediction is for informational purposes only")
with col3:
    st.success("❤️ **Fact:** Healthy lifestyle choices can prevent 80% of heart diseases")