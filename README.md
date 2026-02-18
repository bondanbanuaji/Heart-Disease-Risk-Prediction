# ❤️ Advanced Heart Disease Risk Predictor using C4.5 Decision Tree

This project implements an advanced C4.5 decision tree model to predict the risk of heart disease based on several health-related features. The model is trained on the `heart.csv` dataset and deployed as a modern, interactive web application using Streamlit with enhanced visualizations and user experience.
# ❤️ HeartWise AI: Advanced Risk Predictor

HeartWise AI is a premium, machine-learning-powered application designed to assess cardiovascular risk factors. Utilizing a **C4.5 Decision Tree algorithm**, it provides users with personalized health insights through a modern, responsive interface.

## 🌟 Key Features

- **AI-Powered Prediction**: Accurate risk assessment based on key health parameters.
- **Glassmorphism UI**: A stunning, modern interface with smooth gradients and glass-like components.
- **Interactive Visualizations**: Real-time health fingerprinting via Plotly radar and bar charts.
- **Model Transparency**: Explore the underlying Decision Tree logic directly in the app.
- **Modular Codebase**: Refactored logic for high maintainability and performance.

## 🛠️ Technology Stack

- **Core**: Python
- **Framework**: Streamlit
- **ML Engine**: Scikit-learn (C4.5 Decision Tree)
- **Data Viz**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, Numpy

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher.

### 2. Setup
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone https://github.com/bondanbanuaji/Heart-Disease-Risk-Prediction.git
cd Heart-Disease-Risk-Prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Application
Ensure your virtual environment is active:
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main application logic with modular UI components.
- `heart.csv`: Dataset used for training and logic verification.
- `C45_pinjam_mod.pkl`: Pre-trained C4.5 Decision Tree model.
- `requirements.txt`: Project dependencies.
- `DecisionTree_231351030.ipynb`: Detailed notebook of the training process.

## 🔬 Model Information

The core of the application is a C4.5 Decision Tree. This model was chosen for its interpretability, allowing both clinicians and patients to understand the "why" behind every prediction. Key features include Age, Cholesterol, Blood Pressure, Smoking Habits, Physical Activity, and BMI.

---
**Disclaimer**: HeartWise AI is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

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
