# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:46:39 2023

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the model

load_model = pickle.load(open('C:/Users/Admin/Desktop/P265/Final Model/deployment model/trained_model.sav', 'rb'))

def co2_emission(input_data):
    # Changing the input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we want to predict one term
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = load_model.predict(input_data_reshaped)
    # print(prediction)
    return prediction

def main():
    # Giving a title
    st.title("CO2 emission app")

    # Code for Input data from user
    cylinders = st.text_input("No of cylinder")
    fuel_consumption_comb = st.text_input("No of fuel_consumption_comb(mpg)")
    transmission_Automatedmanual = st.text_input("No of transmission_Automated manual")
    transmission_Automatic = st.text_input("No of transmission_Automatic")
    transmission_Continuouslyvariable = st.text_input("No of transmission_Continuously variable")
    transmission_Manual = st.text_input("No of transmission_Manual")
    fuel_type_Ethanol = st.text_input("No of fuel_type_Ethanol (E85)")
    fuel_type_Naturalgas = st.text_input("No of fuel_type_Natural gas")
    fuel_type_Premiumgasoline = st.text_input("No of fuel_type_Premium gasoline")
    make_1_Luxury = st.text_input("No of make_1_Luxury")
    make_1_Premium = st.text_input("No of make_1_Premium")
    make_1_Sports = st.text_input("No of make_1_Sports")
    Vehicle_Class_Type_sedan = st.text_input("No of Vehicle_Class_Type_sedan")
    Vehicle_Class_Type_suv = st.text_input("No of Vehicle_Class_Type_suv")
    Vehicle_Class_Type_truck = st.text_input("No of Vehicle_Class_Type_truck")

    # Code for prediction
    Co2_calculator = " "

    # Create a button
    if st.button("Co2_emission_value"):
        Co2_calculator = co2_emission([cylinders, fuel_consumption_comb, transmission_Automatedmanual,
                                       transmission_Automatic, transmission_Continuouslyvariable,
                                       transmission_Manual, fuel_type_Ethanol, fuel_type_Naturalgas,
                                       fuel_type_Premiumgasoline, make_1_Luxury, make_1_Premium,
                                       make_1_Sports, Vehicle_Class_Type_sedan, Vehicle_Class_Type_suv,
                                       Vehicle_Class_Type_truck])

    st.success(Co2_calculator)

if __name__ == "__main__":
    main()
