import streamlit as st
import pandas as pd
import joblib

st.title('Fraud Detection Prediction (XGBoost)')

st.markdown('Enter transaction details and press Predict')

model = joblib.load('Fraud_detection_pipeline.pkl')

st.divider()

transcation_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input('Old Balance (Sender)', min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input('New Balance (Sender)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=10000.0)
newbalanceDest = st.number_input('New Balance (Receiver)', min_value=0.0, value=9000.0)


if st.button('Predict'):
    input_data = pd.DataFrame([{
    'type': transcation_type,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest
    }])
    prediction = model.predict(input_data)
    pred_int = int(prediction[0])
    st.subheader(f"Prediction : '{pred_int}'")
    if pred_int == 1:
        st.error('May be Fraud')
    else:
        st.success('Not likely fraud')