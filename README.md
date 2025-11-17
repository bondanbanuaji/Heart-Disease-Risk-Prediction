# ❤️ Advanced Heart Disease Risk Predictor using C4.5 Decision Tree

This project implements an advanced C4.5 decision tree model to predict the risk of heart disease based on several health-related features. The model is trained on the `heart.csv` dataset and deployed as a modern, interactive web application using Streamlit with enhanced visualizations and user experience.

## ✨ Features

- **Intuitive Dashboard**: Modern UI with tab-based navigation
- **Interactive Visualizations**: Plotly charts for comprehensive data analysis
- **Risk Assessment**: Color-coded risk levels with confidence scores
- **Model Insights**: Detailed information about the decision tree model
- **Responsive Design**: Works on both desktop and mobile devices
- **Real-time Predictions**: Instant results based on user inputs

## 📁 Project Structure

```
├── DecisionTree_231351030.ipynb     # Jupyter Notebook with model training & evaluation
├── heart.csv                        # Dataset for model training
├── app.py                          # Enhanced Streamlit web application
├── C45_pinjam_mod.pkl              # Saved trained decision tree model
├── README.md                       # Project documentation
└── LICENSE                         # Project license
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Installation

1. **Clone the repository or download the project files.**

2. **Install the required Python libraries:**

    ```bash
    pip install streamlit pandas scikit-learn matplotlib plotly seaborn numpy
    ```

### Running the Application

1. **Ensure you have the `C45_pinjam_mod.pkl` model file in the same directory as `app.py`.**

2. **Run the Streamlit application from your terminal:**

    ```bash
    streamlit run app.py
    ```

3. **Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).**

## 🎯 How to Use the App

The web application provides an intuitive interface to get a heart disease risk prediction.

### Dashboard Tab
1. **Select your health parameters** from the intuitive form:
   - 👤 Age Group (Senior, Middle-aged, Young)
   - 💉 Cholesterol Level (High, Normal, Low)
   - 🩸 Blood Pressure (High, Normal, Low)
   - 🚬 Smoking Habit (Yes, No)
   - 🏃 Physical Activity (Low, Moderate, High)
   - ⚖️ BMI Category (Overweight, Normal, Obese)

2. **Click the "🔮 Predict Risk Level" button**

3. **View the prediction result** with animated cards showing risk level

### Analysis Tab
- **Interactive Bar Chart**: Visualizes risk factors for each parameter
- **Radar Chart**: Comprehensive view of all health parameters
- **Probability Scores**: Confidence levels for predictions

### Model Insights Tab
- **Decision Tree Visualization**: Interactive tree structure
- **Feature Importance**: Understanding key risk factors
- **Educational Content**: Information about heart disease risk factors

## 📊 Model Information

### Algorithm
- **C4.5 Decision Tree**: Uses information gain for feature selection
- **Features**: Age, Cholesterol, Blood Pressure, Smoking, Physical Activity, BMI
- **Classes**: High Risk, Low Risk, Medium Risk

### Model Performance
The model has been trained and validated on the heart disease dataset with focus on interpretability and accuracy.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Bondan Banuaji** - 231351030

Project Link: [https://github.com/bondanml/heart-disease-predictor](https://github.com/bondanml/heart-disease-predictor)

## 🙏 Acknowledgments

- The heart disease dataset used for training
- Streamlit for the web application framework
- Scikit-learn for the machine learning algorithms
- Plotly for interactive visualizations
