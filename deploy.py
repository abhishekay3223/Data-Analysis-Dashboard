!pip install matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.plotting import scatter_matrix
import warnings
import seaborn as sns
from sklearn.cluster import KMeans
import os
st.set_page_config(page_title="Sales Dashboard",page_icon= "sales.png", layout="wide")
st.title(":bar_chart: Sales Dashboard")
#st.header('E-Commerce Sales Dashboard', divider='rainbow')
#st.header('_Streamlit_ is :blue[cool] :sunglasses:')


#created a dataframe
data = pd.read_csv("Store Data.csv")

#Configuring the page

data.rename(columns={'Age Group': 'Age_Group', 'ship-state': 'ship_state'}, inplace=True)
#sidebar
import streamlit as st
import pandas as pd
# Sidebar

##############

############


st.sidebar.header("Age Filters:")
age_filter_on = st.sidebar.checkbox("Enable Age Filter")

if age_filter_on:
    st.sidebar.subheader("Select Age of Data:")
    min_age = int(data['Age'].min())
    max_age = int(data['Age'].max())
    selected_age = st.sidebar.slider("Select Age:", min_value=min_age, max_value=max_age, value=min_age)
    
    # Filter data based on selected age
    filtered_data = data[data['Age'] == selected_age]
else:
    filtered_data = data
st.sidebar.header("Apply filters:")
# Display filtered data
if st.checkbox("Show Filtered Data"):
    st.write("Filtered Data:")
    st.write(filtered_data)
data=filtered_data



agegroup = st.sidebar.multiselect(
    "Select the Age Group:",
    options = data["Age_Group"].unique(),
    default = data["Age_Group"].unique()
)



gender = st.sidebar.multiselect(
    "Select the Gender:",
    options = data["Gender"].unique(),
    default = data["Gender"].unique()
)
month = st.sidebar.multiselect(
    "Select the Month:",
    options = data["Month"].unique(),
    default = data["Month"].unique()
)
shipstate = st.sidebar.multiselect(
    "Select the Channel:",
    options = data["ship_state"].unique(),
    default = data["ship_state"].unique()
)
# Assuming `data` is your DataFrame


df_selection = data.query("Age_Group == @agegroup & Gender == @gender & Month == @month & ship_state == @shipstate")


if st.checkbox("Show Data"):
   st.dataframe(data)



st.markdown("##")

#top KPI's
total_sales = int(df_selection["Amount"].sum())
total_city = int(len(df_selection["ship-city"].unique()))
total_orders = int(len(df_selection["Qty"]))

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"INR  {total_sales:,}")
with middle_column:
    st.subheader("Cities Delievered:")
    st.subheader(f"  {total_city:,}")
with right_column:
    st.subheader("Total Orders:")
    st.subheader(f"  {total_orders:,}")

st.markdown("---")


import streamlit as st
import pandas as pd

# Assuming your data is loaded into a Pandas DataFrame named 'data'

import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming your data is loaded into a Pandas DataFrame named 'data'

numeric_variables = ['Age', 'Amount', 'Qty']
cols = st.columns(len(numeric_variables))  # Create columns based on number of variables

for i, var in enumerate(numeric_variables):
    checkbox_var = st.sidebar.checkbox(f"Show Histogram for {var}", value=True)

    if checkbox_var:
        with cols[i]:  # Use the corresponding column for each variable
            # Distribution plot using Plotly Express with st.plotly_chart
            fig = px.histogram(data[var], title=f"Histogram of {var}")
            st.plotly_chart(fig)

import streamlit as st
import pandas as pd

# Assuming your data is loaded into a Pandas DataFrame named 'data'

categorical_variables = ['Gender', 'Age_Group', 'Status', 'Channel ', 'Category']

# Function to create a bar chart for a categorical variable
def create_bar_chart(var):
    return st.bar_chart(data[var].value_counts())

# Dropdown menu setup
selected_var = st.sidebar.selectbox("Select Variable for Frequency Distribution:", categorical_variables)

# Display chart based on selected variable
with st.expander(f"Frequency Distribution of {selected_var}"):
    create_bar_chart(selected_var)




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Only imported for compatibility (consider alternatives)

# Assuming your data is loaded into a Pandas DataFrame named 'data'

categorical_variables = ['Gender', 'Age_Group', 'Status', 'Channel ', 'Category']

# Function to create a pie chart for a categorical variable
def create_pie_chart(var):
    fig, ax = plt.subplots(figsize=(5, 5))  # Set appropriate figure size
    data[var].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title(f'Percentage Distribution of {var}')
    return fig

# Dropdown menu setup
selected_var = st.sidebar.selectbox("Select Variable for Percentage Distribution:", categorical_variables)

# Display chart based on selected variable
col1, col2 = st.columns([1, 2])  # Create columns for layout
with col1:
    fig = create_pie_chart(selected_var)
    st.pyplot(fig)


import streamlit as st
import pandas as pd
import seaborn as sns  # Import seaborn for grouped bar chart creation
import matplotlib.pyplot as plt  # Only imported for compatibility (consider alternatives)

# Assuming your data is loaded into a Pandas DataFrame named 'data'

categorical_variables = ['Gender', 'Age_Group', 'Status', 'Channel ', 'Category']

# Function to create a grouped bar chart for a categorical variable
def create_grouped_bar_chart(var):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    sns.countplot(data=data, x=var, hue='Age_Group', ax=ax)  # Use seaborn for the chart
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.set_title(f'Grouped bar chart of {var} by Age Group')

    # Provide a list of tick positions (adjust based on your data)
    tick_positions = [i for i, _ in enumerate(data[var].unique())]  # Assuming unique categories

    # Set x-axis ticks with positions and optional rotation
    ax.set_xticks(tick_positions, labels=data[var].unique(), rotation=45)

    ax.legend(title='Age Group')
    return fig

# Dropdown menu setup
selected_var = st.sidebar.selectbox("Select Variable for Grouped Bar Chart(Age_Group):", options=categorical_variables)

# Conditional display based on selection (using 'if' statement for clarity)
if selected_var:
    fig = create_grouped_bar_chart(selected_var)

    # CSS styling (optional)
    st.markdown("""
    <style>
        .element {
            width: 50%;  /* Adjust width as needed */
            float: left;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.pyplot(fig)  # Remove the 'class_' argument
        st.write('<div class="element"></div>', unsafe_allow_html=True)  # Apply CSS class separately


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_forecast_plot(data):
    # Plot the original time series data
    st.subheader("Original Time Series Data")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(data['Date'], data['Amount'])
    ax.set_title('Original Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    st.pyplot(fig)

    # Check ACF and PACF plots to determine ARIMA parameters
    #st.subheader("Autocorrelation Function (ACF)")
    #fig, ax = plt.subplots()
    #plot_acf(data['Amount'], ax=ax)
    #st.pyplot(fig)

    #st.subheader("Partial Autocorrelation Function (PACF)")
    #fig, ax = plt.subplots()
    #plot_pacf(data['Amount'], ax=ax)
    #st.pyplot(fig)

    order = (0,0,6)
    seasonal_order = (0, 0, 2, 12)

    model = ARIMA(data['Amount'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Forecast for the next 12 months
    forecast_steps = 12
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate dates for the forecast period
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps + 1, freq='M')[1:]

    # Plot the actual data and forecast
    st.subheader("Time-series Forecasting with ARIMA")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(data['Date'], data['Amount'], label='Actual')
    ax.plot(forecast_dates, forecast, label='Forecast', color='red')
    ax.set_title('Time-series Forecasting with ARIMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.legend()
    st.pyplot(fig)

    # Display forecasted values
    st.subheader("Forecasted Values:")
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Amount': forecast})
    st.write(forecast_df)

    

    # Calculate error metrics
    actual_values = data['Amount'].tail(12)
    mae = mean_absolute_error(actual_values, forecast)
    mse = mean_squared_error(actual_values, forecast)
    rmse = np.sqrt(mse)
    st.subheader("Error Metrics:")
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)

    # Calculate accuracy percentage for MSE
    accuracy_mse = round(100 * (1 - (mse / np.mean(np.square(actual_values - actual_values.mean())))), 2)
    st.write("Accuracy percentage (MSE):", accuracy_mse, "%")

# Sample data generation (replace this with your actual data)
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
data = pd.DataFrame({'Date': dates, 'Amount': np.random.randn(len(dates))})

# Render the forecast plot in Streamlit
st.title("ARIMA Time-series Forecasting")
generate_forecast_plot(data)





