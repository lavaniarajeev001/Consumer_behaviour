import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data():
    df = pd.read_csv("Customer Purchasing Behaviors.csv")
    df.region = df.region.map({"East": 1, "West": 2, "North": 3, "South": 4})
    df = df.drop(["user_id"], axis=1)
    return df

def add_prediction(input_data):
    # Load the trained model
    with open("model.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
    
    # Convert input data to array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Load the scaler used during training
    with open("scaler.pkl", "rb") as scaler_in:
        scaler = pickle.load(scaler_in)
    
    # Transform the input data using the loaded scaler
    input_scaled = scaler.transform(input_array)
    prediction = classifier.predict(input_scaled)

    st.subheader("Prediction")
    st.write("The person is:")  
    
    if prediction < 4:
        st.write("Consumer is not buying")
    elif 4 <= prediction < 7:
        st.write("Consumer may buy or not buy")
    else:
        st.write("Consumer will buy")

def add_sidebar():
    st.sidebar.header("Consumer attributes")
    df = data()
    
    slider_labels = [
        ("age", "age"),
        ("annual_income", "annual_income"),
        ("purchase_amount", "purchase_amount"),
        ("region(East=1,West=2,North=3,South=4", "region"),
        ("purchase_frequency", "purchase_frequency")
    ]
    
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(df[key].max())
        )
    return input_dict

def main():
    st.set_page_config(
        page_title="Consumer Behaviour Prediction",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Consumer Behaviour App")
        st.write("This app is designed for the prediction of customer behavior based on the provided attributes.")
    
    if st.button("Predict"):
        add_prediction(input_data)

if __name__ == "__main__":
    main()
