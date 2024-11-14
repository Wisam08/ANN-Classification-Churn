import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

model= tf.keras.models.load_model('model.h5')

##load the encoder and scalar
with open('onehot_encoder_geo.pk1','rb') as file:
    label_encoder_geo = pickle.load(file)

## load the rest
with open('label_encoder_gender.pk1','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pk1','rb') as file:
    scalar = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    #'Geography': [label_encoder_geo.transform([geography])[0]],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine the one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)

#Scale te=he input data
input_data_scaled = scalar.transform(input_data)

prediction = model.predict(input_data_scaled)
pred_prob=prediction[0][0]

st.write(f'Churn Probability: {prediction[0][0]:.2f}')

if(prediction[0][0] > 0.5):
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn")
