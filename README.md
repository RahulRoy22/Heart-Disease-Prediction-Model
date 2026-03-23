# 🫀 Heart Disease Prediction Model - Advanced ML System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced machine learning system for predicting heart disease using ensemble methods, deep learning, and comprehensive feature engineering. This project demonstrates state-of-the-art ML techniques including stacking ensembles, neural networks, and production-ready pipelines.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project predicts the presence of heart disease in patients based on 13 clinical features. It implements multiple machine learning algorithms and compares their performance to identify the best model for deployment.

### Key Highlights:
- **6 Different ML Models** including ensemble and deep learning approaches
- **5 Evaluation Metrics** for comprehensive performance analysis
- **Feature Engineering** with StandardScaler normalization
- **Hyperparameter Tuning** using GridSearchCV
- **Model Persistence** for production deployment
- **Prediction Pipeline** with confidence scores

## ✨ Features

### Machine Learning Models
1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble decision trees
3. **Gradient Boosting** - Sequential ensemble learning
4. **Support Vector Machine (SVM)** - Kernel-based classifier
5. **Stacking Ensemble** - Meta-learning combining RF, GB, and SVM
6. **Neural Network** - Deep learning with 4 layers and dropout

### Advanced Techniques
- ✅ Feature scaling with StandardScaler
- ✅ Cross-validation (5-fold)
- ✅ Hyperparameter optimization
- ✅ Ensemble stacking
- ✅ Deep learning with regularization
- ✅ Feature importance analysis
- ✅ Confusion matrix visualization
- ✅ Model persistence (save/load)

### Evaluation Metrics
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve

## 📊 Dataset

The dataset contains **303 patient records** with **14 attributes**:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Gender (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-4) | Numeric |
| thal | Thalassemia (0-3) | Categorical |
| **target** | **Heart disease (1=yes, 0=no)** | **Binary** |

### Class Distribution
- **Positive (Disease)**: 165 samples (54.5%)
- **Negative (Healthy)**: 138 samples (45.5%)

## 🤖 Models

### 1. Stacking Ensemble (Recommended)
Combines predictions from multiple base models:
- **Base Models**: Random Forest, Gradient Boosting, SVM
- **Meta-Learner**: Logistic Regression
- **Expected Accuracy**: 85-92%
- **Expected ROC-AUC**: 0.90-0.95

### 2. Neural Network
Deep learning architecture:
```
Input (13 features)
    ↓
Dense(64, ReLU) → Dropout(0.3)
    ↓
Dense(32, ReLU) → Dropout(0.3)
    ↓
Dense(16, ReLU)
    ↓
Dense(1, Sigmoid)
```
- **Training**: 100 epochs with validation split
- **Regularization**: Dropout layers to prevent overfitting

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Fix Protobuf Conflicts (if needed)
```bash
pip install --upgrade protobuf==4.25.3
```

## 💻 Usage

### Running the Notebook

#### Option 1: Jupyter Notebook
```bash
jupyter notebook Heart_Disease_Prediction_Model_Upgraded.ipynb
```

#### Option 2: Google Colab
1. Upload the notebook to Google Colab
2. Upload `data.csv` to Colab
3. Run all cells sequentially

### Making Predictions

After training the models, use the prediction pipeline:

```python
# Example patient data
# Format: (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
patient_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)

# Make prediction with stacking ensemble
result, confidence = predict_heart_disease(patient_data, model_type='stacking')
print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2%}")
```

### Loading Saved Models

```python
import joblib
from tensorflow import keras

# Load models
stacking_model = joblib.load('heart_disease_stacking_model.pkl')
scaler = joblib.load('heart_disease_scaler.pkl')
nn_model = keras.models.load_model('heart_disease_nn_model.h5')

# Make prediction
input_scaled = scaler.transform([patient_data])
prediction = stacking_model.predict(input_scaled)
```

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~85% | ~0.85 | ~0.87 | ~0.86 | ~0.90 |
| Random Forest | ~87% | ~0.88 | ~0.89 | ~0.88 | ~0.92 |
| Gradient Boosting | ~88% | ~0.89 | ~0.90 | ~0.89 | ~0.93 |
| SVM | ~86% | ~0.87 | ~0.88 | ~0.87 | ~0.91 |
| **Stacking Ensemble** | **~90%** | **~0.91** | **~0.92** | **~0.91** | **~0.95** |
| Neural Network | ~89% | ~0.90 | ~0.91 | ~0.90 | ~0.94 |

*Note: Actual results may vary based on random seed and data split*

### Key Insights
- **Best Model**: Stacking Ensemble (combines strengths of multiple models)
- **Most Important Features**: cp (chest pain), thalach (max heart rate), oldpeak
- **Improvement over Baseline**: 8-10% accuracy increase from original Logistic Regression

## 📁 Project Structure

```
Heart-Disease-Prediction-Model/
│
├── Heart_Disease_Prediction_Model.ipynb          # Original baseline model
├── Heart_Disease_Prediction_Model_Upgraded.ipynb # Advanced upgraded model ⭐
├── data.csv                                       # Dataset
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore rules
├── README.md                                      # This file
│
└── Generated Files (after running notebook):
    ├── heart_disease_stacking_model.pkl          # Saved stacking model
    ├── heart_disease_scaler.pkl                  # Saved feature scaler
    └── heart_disease_nn_model.h5                 # Saved neural network
```

## 🔬 Methodology

### 1. Data Preprocessing
- Check for missing values (none found)
- Analyze class distribution (relatively balanced)
- Feature correlation analysis

### 2. Feature Engineering
- StandardScaler normalization (mean=0, std=1)
- Preserves feature relationships while scaling

### 3. Model Training
- Train-test split: 80-20 with stratification
- Cross-validation for robust evaluation
- Hyperparameter tuning with GridSearchCV

### 4. Ensemble Learning
- Stacking combines diverse models
- Meta-learner learns optimal combination
- Reduces overfitting and improves generalization

### 5. Evaluation
- Multiple metrics for comprehensive assessment
- Confusion matrix for error analysis
- Feature importance for interpretability

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning
- **Joblib** - Model persistence

## 📝 Future Enhancements

- [ ] SMOTE for handling class imbalance
- [ ] XGBoost/LightGBM implementation
- [ ] SHAP values for explainable AI
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Docker containerization
- [ ] Real-time prediction dashboard
- [ ] Automated retraining pipeline
- [ ] A/B testing framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Inspiration from various Kaggle kernels
- scikit-learn and TensorFlow documentation

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com)

---

⭐ If you found this project helpful, please give it a star!

**Made with ❤️ and Python**
