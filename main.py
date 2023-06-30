import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('data_clean.csv')
df_train = pd.read_csv('X_train_clean.csv')
num_cols = ['completion_year', 'parking', 'bathroom', 'size_sqft', 'rooms_num']
cols = df_train.columns

def plot_size_by_rooms():
    data = df.groupby('rooms_num')['size_sqft'].median()
    rc = {'figure.figsize':(2,2),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 4,
          'xtick.labelsize': 4,
          'ytick.labelsize': 4}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=data.index, y=data.values, data=data, color = '#b80606')
    ax.set(xlabel = 'Number of Rooms', ylabel = 'Property Size (sq.ft.)')
    st.pyplot(fig, ax)

def plot_price_by_size():
    rc = {'figure.figsize':(4,4),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 8,
          'xtick.labelsize': 5,
          'ytick.labelsize': 5}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.regplot(x=df['size_sqft'], y=df['monthly_rent_rm'], x_jitter=.1, data=df, color = '#b80606',scatter_kws={"color": "#b80606", 's':3},line_kws={"color": "#ed8e8e", 'alpha':0.7, 'linewidth':2})
    ax.set(xlabel = 'Size (sq.ft.)', ylabel = 'Monthly Rent (RM)')
    st.pyplot(fig, ax)

def plot_price_by_furnished():
    rc = {'figure.figsize':(4,4),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 8,
          'xtick.labelsize': 5,
          'ytick.labelsize': 5}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.boxplot(x=df['furnished'], y=df['monthly_rent_rm'], data=df, color = '#b80606', fliersize=3)
    ax.set(xlabel = 'Property Size (sq.ft.)', ylabel = 'Monthly Rent (RM)')
    st.pyplot(fig, ax)

def plot_price_by_nearby_railways():
    rc = {'figure.figsize':(4,4),
            'axes.facecolor':'#0e1117',
            'axes.edgecolor': '#0e1117',
            'axes.labelcolor': 'white',
            'figure.facecolor': '#0e1117',
            'patch.edgecolor': '#0e1117',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': 'grey',
            'font.size' : 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 5,
            'ytick.labelsize': 5}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.boxplot(x=df['nearby_railways'], y=df['monthly_rent_rm'], data=df, color = '#b80606', fliersize=3)
    ax.set(xlabel = 'Property is nearby to railway station (KTM/LRT)', ylabel = 'Monthly Rent (RM)')
    st.pyplot(fig, ax)

def plot_price_per_sqft_by_area():
    data = df.groupby('area').apply(lambda x: x.monthly_rent_rm.sum() / x.size_sqft.sum()).sort_values(ascending=False)

    rc = {'figure.figsize':(8,20),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ax = sns.barplot(x=data.values, y=data.index, orient='h', color = '#b80606')
    ax.set(xlabel = 'Mean Price per sqft (RM/sqft)', ylabel = 'Property Area')
    st.pyplot(fig, ax)
    

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
    col_name = "furnished_" + furnished
    return col_name

def getStandardized(col, val):
    mean = dict_means[col]
    std_dev = dict_std_dev[col]
    return (val - mean) / std_dev

@st.cache_resource
def load_model(model):
    return joblib.load(model)

dict_model = {
    'Linear Regression': 'model/lin_reg_model.joblib',
    'Gradient Boost': 'model/grad_model.joblib',
    'Random Forest': 'model/rf_model.joblib',
}

def get_user_inputs():
    location_options = sorted(df['area'].unique())
    property_type_options = sorted(df['property_type'].unique())

    reg = st.selectbox("Regression Model", list(dict_model.keys()))
    location = st.selectbox("Location", location_options)
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

    model = load_model(dict_model[reg])
    df_input = pd.DataFrame(data=np.zeros(shape=(1, len(df_train.columns))), columns=model.feature_names_in_, dtype='float')
    df_input[convert_location(location)] = 1.0
    df_input['completion_year'] = getStandardized('completion_year', year)
    df_input['size_sqft'] = getStandardized('size_sqft', size)
    df_input['rooms_num'] = getStandardized('rooms_num', rooms)
    df_input['bathroom'] = getStandardized('bathroom', bathrooms)
    df_input['parking'] = getStandardized('parking', parking)
    df_input[[convert_furnished(furnished)]] = 1.0
    df_input['nearby_railways_yes'] = convert_nearby_railways(nearby_railways)

    return model, df_input

st.set_page_config(layout="wide")
st.title("Predicting your monthly rental price")
col1, col2 = st.columns(2,gap='medium')
with col1: 
    plot_size_by_rooms()
    plot_price_by_size()
    plot_price_by_furnished()
with col2:
    plot_price_per_sqft_by_area()
    plot_price_by_nearby_railways()
st.divider()
model, user_input = get_user_inputs()
st.divider()
pred = f"RM{model.predict(user_input)[0]:.0f}"
st.metric(label="The monthly rental price of this propert is estimated to be", value=pred)



