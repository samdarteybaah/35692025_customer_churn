import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

#loading the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

#loading the model
tuned_model = load_model('tuned_model.h5')

# title for the model 
st.title("Customer Churn Prediction Web App")

st.header("User Input Features")

# creating the inputs on the web app
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=1.0, value=0.0)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)

tenure = st.number_input("Tenure", min_value=0, max_value=150, step=1, value=0)

contract_encoded_mapping = {0: 'Month-to-month', 1: 'Two year', 2: 'One year'}
contract_encoded = st.selectbox("Contract", list(contract_encoded_mapping.keys()), format_func=lambda x: contract_encoded_mapping[x])

PaymentMethod_encoded_mapping = {0: 'Electronic check', 1: 'Mailed check', 2: 'Bank transfer (automatic)', 3: 'Credit card (automatic)'}
PaymentMethod_encoded = st.selectbox("Payment Method", list(PaymentMethod_encoded_mapping.keys()), format_func=lambda x: PaymentMethod_encoded_mapping[x])

InternetService_encoded_mapping = {0: 'DSL', 1: 'Fiber optic', 2: 'No'}
InternetService_encoded = st.selectbox("Internet service", list(InternetService_encoded_mapping.keys()), format_func=lambda x: InternetService_encoded_mapping[x])

#creating a dataframe
if st.button("Submit"): 
    user_input = pd.DataFrame({
        'TotalCharges': [total_charges],
        'MonthlyCharges': [monthly_charges],
        'tenure': [tenure],
        'Contract_encoded': [contract_encoded],
        'PaymentMethod_encoded': [PaymentMethod_encoded],
        'InternetService_encoded': [InternetService_encoded]
    })

    # scaling the data input (scale it)
    StandardScaler = loaded_scaler
    needed_features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract_encoded', 'PaymentMethod_encoded', 'InternetService_encoded']
    user_input_scaled = StandardScaler.transform(user_input[needed_features])
    user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=needed_features)

    #  using the tuned model to make predictions
    prediction = tuned_model.predict(user_input_scaled_df)

    # displaying the prediction
    st.subheader("Prediction")
    churn_status = "This customer's status shows they are likely to churn" if prediction[0, 0] > 0.5 else "This customer's status shows they are not likely to churn"
    confidence_score = prediction[0, 0]
    st.write(f"The predicted churn status is: {churn_status}")
    st.write(f"Confidence Score: {confidence_score:.2%}")



st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px #888888;
        }
        .main .block-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px #888888;
        }
    </style>
    """,
    unsafe_allow_html=True
)