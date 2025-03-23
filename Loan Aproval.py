# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib

# Load the CSV file into a DataFrame
df = pd.read_csv('loan_data.csv')

# Display the first few rows
print(df.head())


# Check dataset info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())


# Handle missing values
def preprocess_data(df):
    # Drop Loan_ID column
    df = df.drop(columns=['Loan_ID'], errors='ignore')
    
    # Convert 'Dependents' to numerical
    df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(float)
    
    # Fill missing values
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].median())
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    
    # Feature Engineering
    df['TotalIncome'] = np.log1p(df['ApplicantIncome'] + df['CoapplicantIncome'])
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['BalanceIncome'] = df['TotalIncome'] - df['EMI']
    
    return df

df = preprocess_data(df)

# Check for missing values
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Define features and target
X = df.drop(columns=['Loan_Status', 'ApplicantIncome', 'CoapplicantIncome'])
y = df['Loan_Status'].map({'N': 0, 'Y': 1})

# Identify feature types
numeric_features = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                    'TotalIncome', 'EMI', 'BalanceIncome']
categorical_features = ['Gender', 'Married', 'Dependents', 'Education',
                        'Self_Employed', 'Property_Area']



# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define models with class weights
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
}


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define models with class weights
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
}


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Use only Random Forest
model = make_imb_pipeline(
    preprocessor,
    SMOTE(sampling_strategy='auto', random_state=42),
    RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=5,
        random_state=42
    )
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Feature Importance Extraction
# -----------------------------------------------
try:
    # Get preprocessor and model from pipeline
    preprocessor = model.named_steps['columntransformer']
    rf_model = model.named_steps['randomforestclassifier']
    
    # Get feature names
    numeric_feats = numeric_features
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feats = list(cat_encoder.get_feature_names_out(categorical_features))
    all_features = numeric_feats + cat_feats
    
    # Get importances
    importances = rf_model.feature_importances_
    
    print("\nTop 10 Feature Importances:")
    for feat, imp in sorted(zip(all_features, importances), 
                          key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat}: {imp:.4f}")

except Exception as e:
    print(f"Feature importance error: {str(e)}")

import pickle

# Assuming 'model' is your trained machine learning model
with open("loan_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)



