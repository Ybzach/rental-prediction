import streamlit as st
import pandas as pd
import numpy as np
df = pd.read_csv('data_clean.csv')
df_train = pd.read_csv('X_train.csv')
num_cols = ['completion_year', 'parking', 'bathroom', 'size_sqft', 'rooms_num']
cols = df_train.columns
def getMeans():
    means = {}
    for i in range(len(num_cols)):
        means.update({num_cols[i]: df_train[num_cols[i]].mean()})
    return means

def getStdDevs():
    std_devs = {}
    for i in range(len(num_cols)):
        std_devs.update({num_cols[i]: df_train[num_cols[i]].std()})
    return std_devs

dict_means = getMeans()
dict_std_dev = getStdDevs()

def convert_nearby_railways(bool):
    if (bool):
        return 1.0
    return 0

def convert_location(location):
    col_name = "area_" + location
    return col_name

def convert_property_type(property_type):
    col_name = "property_type_" + property_type
    return col_name

def convert_furnished(furnished):
    col_name = "furnished" + furnished
    return col_name

def getStandardized(col, val):
    mean = dict_means[col]
    std_dev = dict_std_dev[col]
    return (val - mean) / std_dev


st.title("Predicting your monthly rental price")
# def get_user_inputs():
location_options = sorted(df['area'].unique())
property_type_options = sorted(df['property_type'].unique())

location = st.selectbox("Location", location_options)
property_type = st.selectbox("Property Type", property_type_options)
size = st.number_input("Size (sq.ft.)", min_value = 50, max_value = 5000, step=10, value=200)
year = st.slider("Completion Year", 2000, 2023, 2016)
col1, col2 = st.columns(2)
with col1:
    rooms = st.slider("Number of rooms", 0, 10, 3)
    bathrooms = st.slider("Number of bathrooms", 0, 10, 2)
with col2:
    parking = st.slider("Number of parking slots", 0, 10, 2)
    furnished = st.select_slider("Furnishing", ['Not Furnished', 'Partially Furnished', 'Fully Furnished'], 'Partially Furnished')
nearby_railways = st.checkbox("Close proximity to a railway station (KTM/LRT)")

df_input = pd.DataFrame(data=np.zeros(shape=(1, len(df_train.columns))), columns=cols, dtype='float')
df_input[convert_location(location)] = 1.0
df_input[convert_property_type(property_type)] = 1.0
df_input['completion_year'] = year
df_input['size_sqft'] = getStandardized('size_sqft', size)
df_input['rooms_num'] = getStandardized('rooms_num', rooms)
df_input['bathroom'] = getStandardized('bathroom', bathrooms)
df_input['parking'] = getStandardized('parking', parking)
df_input[[convert_furnished(furnished)]] = 1.0
df_input['nearby_railways_yes'] = convert_nearby_railways(nearby_railways)




