# Loan Approval Prediction App 🏦✅  

![Loan Approval](https://i.pinimg.com/736x/a6/8b/79/a68b790991934255bb0790598c45abde.jpg)  

## 🚀 About the Project  
This is a **Machine Learning-powered Loan Approval Prediction App** that predicts whether a loan application will be approved based on various factors. The project is built using **Python, Scikit-learn, Pandas, and NumPy** and deployed on **Render**.  

🔗 **Live Demo:** [Loan Approval Predictor](https://loan-approval-3.onrender.com/)  

---  

## 🛠️ Features  
✔️ Predicts **loan approval** based on multiple applicant features  
✔️ Uses **machine learning (Random Forest Classifier)** for predictions  
✔️ **Handles imbalanced data** using SMOTE  
✔️ **Feature engineering** for better accuracy  
✔️ Deployed on **Render** for seamless access  

---  

## 📊 Data Source  
The dataset used for this project is from **loan application records**.  

- 📂 **Dataset File:** `loan_data.csv`  
- 📑 **Preprocessing Steps:**  
  - Removed **Loan ID** as it’s not relevant  
  - Converted **Dependents** with `"3+"` to `3`  
  - Filled missing values using **mode/median strategies**  
  - Created new features like **Total Income, EMI, and Balance Income**  
  - Standardized numerical features and one-hot encoded categorical features  

---  

## 📌 How It Works  
1️⃣ Enter applicant details like **Gender, Income, Credit History, Loan Amount, etc.**  
2️⃣ Click on **Predict Loan Approval**  
3️⃣ The app runs a **pre-trained ML model** to predict approval status  
4️⃣ Displays result as **Approved ✅ or Not Approved ❌**  

---  

## 🏗️ Tech Stack  
- **Backend:** Python (Flask)  
- **ML Model:** Scikit-learn (Random Forest Classifier)  
- **Data Processing:** Pandas, NumPy  
- **Deployment:** Render  
- **Model Handling:** Joblib & Pickle  

---  

## 🧠 Machine Learning Model  

### 🔹 Preprocessing  
- Handled missing values with **mode and median imputation**  
- Engineered new features (**Total Income, EMI, Balance Income**)  
- Standardized numerical columns  
- Applied **One-Hot Encoding** to categorical features  

### 🔹 Model Training  
We tested multiple models:  
✅ **Logistic Regression** – Good but slightly lower accuracy  
✅ **Support Vector Machine (SVM)** – Performed well but computationally expensive  
✅ **Random Forest Classifier** – 🚀 **Best Model (Final Choice)**  

📌 **Final Model Chosen:** `RandomForestClassifier` with SMOTE for handling class imbalance  

### 🔹 Model Evaluation  
- Used **ROC-AUC Score** and **Confusion Matrix**  
- **Final Accuracy:** *82%*  
- Feature importance extracted to understand key factors in approval  

📂 **Model Saved as:** `loan_model.pkl`  

---  

## 🔧 Installation & Running Locally  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/MK3685/loan-approval-prediction.git
cd loan-approval-prediction
