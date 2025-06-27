# 📊 Machine Learning Projects: Classification & Regression

This repository contains **two end-to-end ML projects** — one for **customer churn prediction** (classification) and the other for **house price prediction** (regression). Both projects include **exploratory data analysis (EDA)**, **feature engineering**, **model training**, **evaluation**, and **model serialization**.

---

## 📁 Project Structure

```
.
├── customer churn predictor (Classification)
│   ├── data/
│   │   └── Customer Churn.csv
│   ├── model files/
│   │   ├── logistic_model.pkl
│   │   ├── random_forest_model.pkl
│   │   └── xgboost_model.pkl
│   └── churn.py
│
├── house price prediction (Regression)
│   ├── data/
│   │   └── housing.csv
│   └── model.py
│
└── README.md
```

---

## 📌 Project 1: Customer Churn Predictor (Classification)

### 📄 Dataset
The dataset contains customer service details including demographics, subscription plans, and churn status.  
**Target:** `Churn` (Yes/No)

### 🧪 ML Models Used
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

### 🔧 Key Steps
- Cleaned missing and invalid entries
- Dropped high-cardinality and irrelevant columns (e.g., `customerID`, `TotalCharges`)
- One-hot encoding of categorical variables
- Feature engineering:
  - `Tenure * MonthlyCharges`
  - `NumServices` subscribed
  - `HighCharges` binary flag
- Model evaluation with:
  - Accuracy
  - Confusion matrix
  - Classification report

### 📦 Output
Trained models saved as:
- `logistic_model.pkl`
- `random_forest_model.pkl`
- `xgboost_model.pkl`

📸 _Sample Visualizations:_
![image](https://github.com/user-attachments/assets/3829d449-9afc-4f46-8d5e-5e71168d1620)

![image](https://github.com/user-attachments/assets/1c1997e1-798c-42dd-b227-54f9d0a7c82e)
![image](https://github.com/user-attachments/assets/84c403c8-44bc-41bb-bf2a-98574a72b6b4)
![image](https://github.com/user-attachments/assets/c53f61e4-e423-47b8-8976-c5d5a00e6b12)
![image](https://github.com/user-attachments/assets/3dac89ec-643b-4ea2-bb3c-08f72d4c9aa2)
![image](https://github.com/user-attachments/assets/00002773-a520-4352-b8a1-23e00433150c)
![image](https://github.com/user-attachments/assets/9a19514e-f1b2-4bf8-a512-7f90497df8cb)
![image](https://github.com/user-attachments/assets/35e4248a-a886-48df-9619-09adbe81c332)
![image](https://github.com/user-attachments/assets/72eddc67-0d83-4c08-93a1-d985c6d117e1)
![image](https://github.com/user-attachments/assets/beab9439-fb06-44b6-937d-bdf6a3a934b1)
![image](https://github.com/user-attachments/assets/22135cdc-33d6-43d7-8243-0eed9323f895)
![image](https://github.com/user-attachments/assets/eaa3c3f9-2579-46ff-ae6c-ff6f40d8932d)
![image](https://github.com/user-attachments/assets/96a7d0a0-0234-4146-a999-2fce96c5169a)
![image](https://github.com/user-attachments/assets/f3d3e081-ddb2-469e-ba4d-655528bd53e9)
![image](https://github.com/user-attachments/assets/9bc4137b-e7ab-498f-b499-75864e7f97b3)
![image](https://github.com/user-attachments/assets/77594e4a-aa3e-4b15-981c-800a060ea46f)
![image](https://github.com/user-attachments/assets/85dd5045-4852-4590-9bf4-2899fe6d793d)

![image](https://github.com/user-attachments/assets/57c2d4cd-13af-446f-89a5-f26929867fc8)

![image](https://github.com/user-attachments/assets/1b8133f8-f40c-4f2f-a9e4-089cbe224b6b)

---

## 📌 Project 2: House Price Prediction (Regression)

### 📄 Dataset
California Housing dataset with features like `median_income`, `housing_median_age`, etc.  
**Target:** `median_house_value`

### 🧪 ML Model Used
- **Linear Regression**

### 🔧 Key Steps
- Dropped columns with excessive missing data (`total_bedrooms`)
- Exploratory data analysis using seaborn and feature binning
- One-hot encoding for `ocean_proximity`
- Scaling using `MinMaxScaler`
- Evaluation using:
  - Mean Squared Error (MSE)
  - R² Score

📸 _Sample Visualizations:_
![image](https://github.com/user-attachments/assets/928a28c8-86c7-4595-b1a7-c5104fd853e7)

![image](https://github.com/user-attachments/assets/4e848fd3-c98f-4f3c-b7f4-bf269eb938b2)
![image](https://github.com/user-attachments/assets/455988e9-f32e-4080-9a54-c955c5ead315)
![image](https://github.com/user-attachments/assets/16cad776-eb5d-45d4-9646-63d06030dbce)
![image](https://github.com/user-attachments/assets/d06eddb3-3717-40d6-b4ed-f17178d78e8e)
![image](https://github.com/user-attachments/assets/f8b882d0-6921-4725-a22a-f19176476930)
![image](https://github.com/user-attachments/assets/fe18b1e8-b3f3-4ed6-829c-c1a688fa9c8e)
![image](https://github.com/user-attachments/assets/e58d371d-6369-42c1-adb3-d875652fa981)

---

## 🧠 Skills Demonstrated

✅ EDA and Visualization  
✅ Data Cleaning & Preprocessing  
✅ Feature Engineering  
✅ Classification and Regression Models  
✅ Model Evaluation & Metrics  
✅ Model Saving (Serialization)  
✅ Code Modularization by Project

---

## 💻 How to Run

1. **Install Dependencies**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib
```

2. **Run Each Script**
```bash
# For churn classification
python churn.py

# For house price regression
python model.py
```

---

## 📬 Contact

Made with 💡 by **Tanishka Nagawade**  
For suggestions or contributions, feel free to open an issue or fork the repo!
