import pandas as pd
import streamlit as st
import sklearn
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
st.set_page_config(layout="wide")
# App Layout
st.title('Boston Houses Price Prediction')
st.image("hu.jpeg")
# loading the model
model = joblib.load('mymodel')
# entering the house features
st.header('Enter the following House Features:')
CRIM = st.number_input("per capita crime rate by town:", min_value=0, max_value=2000, value=0)
ZN = st.number_input("proportion of residential land zoned for lots over 25,000 sq.ft.:", min_value=25000,
                     max_value=200000, value=25000)
INDUS = st.number_input(" proportion of non-retail business acres per town", min_value=0, max_value=2000, value=0)
NOX = st.number_input("nitric oxides concentration (parts per 10 million)", min_value=0, max_value=2000000, value=0)
CHAS = st.number_input("Charles River dummy variable (1 if tract bounds river; 0 otherwise)", min_value=0, max_value=1,
                       value=0)
RM = st.number_input(" average number of rooms per dwelling", min_value=1, max_value=10, value=1)
AGE = st.number_input("proportion of owner-occupied units built prior to 1940", min_value=0, max_value=2000, value=0)
DIS = st.number_input("weighted distances to five Boston employment centres", min_value=0, max_value=2000, value=0)
RAD = st.number_input("index of accessibility to radial highways", min_value=0, max_value=2000, value=0)
TAX = st.number_input("full-value property-tax rate per $10,000:", min_value=0, max_value=2000, value=0)
PTRATIO = st.number_input("pupil-teacher ratio by town:", min_value=0, max_value=200, value=0)
B = st.number_input("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town", min_value=0, max_value=2000,
                    value=0)
LSTAT = st.number_input("% lower status of the population", min_value=0, max_value=2000, value=0)
data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]], columns=
['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
 'PTRATIO', 'B', 'LSTAT'])
scaled_data = pd.DataFrame(scaler.fit_transform(data),
                           columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                    'PTRATIO', 'B', 'LSTAT'])
b = st.button("Predict House Price")
if b:
    st.write("The price is in thousand dollars")
    st.subheader(model.predict(data)[0])