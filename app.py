from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': float(request.form['dependents']),
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'Property_Area': request.form['property_area'],
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': float(request.form['credit_history'])
        }
        
        # Create DataFrame and add engineered features
        df = pd.DataFrame([data])
        df['TotalIncome'] = np.log1p(float(request.form['applicant_income']) + float(request.form['coapplicant_income']))
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        df['BalanceIncome'] = df['TotalIncome'] - df['EMI']
        
        # Drop unnecessary fields
        df = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'Property_Area', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                'TotalIncome', 'EMI', 'BalanceIncome']]
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1][0]
        
        result = {
            'prediction': 'Approved' if prediction[0] == 1 else 'Not Approved',
            'probability': f"{probability*100:.2f}%"
        }
        
        return render_template('after.html', result=result)
    
    except Exception as e:
        return render_template('after.html', 
                             result={'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)