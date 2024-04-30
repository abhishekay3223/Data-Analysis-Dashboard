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
#`data` is your DataFrame


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

###################


#########

import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
# Replace 'data.csv' with the actual path to your data file
# data = pd.read_csv('data.csv')

# Get the list of categorical variables
categorical_variables = ['Gender', 'Age_Group', 'Status', 'Category']

# Function to create a bar chart for a categorical variable using Plotly
def create_bar_chart(var):
    chart_data = data[var].value_counts().reset_index()
    chart_data.columns = [var, 'Count']
    fig = px.bar(chart_data, x=var, y='Count', labels={var: var, 'Count': 'Count'})
    st.plotly_chart(fig)

# Function to create a pie chart for a categorical variable using Plotly
def create_pie_chart(var):
    chart_data = data[var].value_counts().reset_index()
    chart_data.columns = [var, 'Count']
    fig = px.pie(chart_data, values='Count', names=var, title=f'Percentage Distribution of {var}')
    st.plotly_chart(fig)

# Function to create a grouped bar chart for a categorical variable using Plotly
def create_grouped_bar_chart(var):
    grouped_data = data.groupby([var, 'Age_Group']).size().reset_index(name='Count')
    fig = px.bar(grouped_data, x=var, y='Count', color='Age_Group', barmode='group',
                 labels={var: var, 'Count': 'Count', 'Age_Group': 'Age Group'})
    st.plotly_chart(fig)

# Layout setup
st.title("Data Analysis")

# Create a row layout for dropdown menus
col1, col2, col3 = st.columns(3)

# Dropdown menu for selecting categorical variables
with col1:
    selected_var1 = st.selectbox("Frequency Distribution:", categorical_variables)
    create_bar_chart(selected_var1)

with col2:
    selected_var2 = st.selectbox("Percentage Distribution:", categorical_variables)
    create_pie_chart(selected_var2)

with col3:
    categorical_variables = ['Gender', 'Status', 'Category']
    selected_var3 = st.selectbox("Grouped Bar Chart:", categorical_variables)
    create_grouped_bar_chart(selected_var3)

#########

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Function to plot the histogram for top states using Plotly
def plot_top_states_histogram(num_states):
    # Get the top states with the highest number of orders
    top_states = data['ship_state'].value_counts().nlargest(num_states)

    # Plotting the histogram using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_states.index, y=top_states.values))

    # Annotate each bar with its order count
    for i, count in enumerate(top_states):
        fig.add_annotation(
            x=top_states.index[i],
            y=count + 1,
            text=str(count),
            showarrow=False,
            font=dict(color='black', size=12),
            xanchor='center',
            yanchor='bottom'
        )

    fig.update_layout(
        title=f'Top {num_states} States with Highest Number of Orders',
        xaxis_title='State',
        yaxis_title='Number of Orders',
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig)

# Function to plot the histogram for top cities using Plotly
def plot_top_cities_histogram(selected_state):
    # Filter data for the selected state
    state_data = data[data['ship_state'] == selected_state]

    # Group the data by 'ship-city' and count the orders for each city
    top_cities = state_data['ship-city'].value_counts().nlargest(5)

    # Plotting the histogram using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top_cities.index, y=top_cities.values))

    fig.update_layout(
        title=f'Top 5 Cities with Highest Number of Orders in {selected_state}',
        xaxis_title='Ship City',
        yaxis_title='Number of Orders',
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig)

# Streamlit UI
st.title('Top States and Cities with Highest Number of Orders')

# Slider to select the number of states
num_states = st.slider('Select Number of States', min_value=1, max_value=50, value=10, step=1)

# Dropdown menu for selecting ship state
selected_state = st.selectbox('Select a Ship State:', data['ship_state'].unique())

# Plot both charts in a single row
#st.subheader("Top States with Highest Number of Orders")
#st.subheader(f"Top 5 Cities with Highest Number of Orders in {selected_state}")
col1, col2 = st.columns(2)
with col1:
    plot_top_states_histogram(num_states)
with col2:
    plot_top_cities_histogram(selected_state)


#########



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_forecast_plot(data):
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
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps + 1, freq='MS')[1:]
    

    

    

# Sample data generation (replace this with your actual data)
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='MS')
data = pd.DataFrame({'Date': dates, 'Amount': np.random.randn(len(dates))})

# Render the forecast plot in Streamlit
st.title("ARIMA Time-series Forecasting")
generate_forecast_plot(data)

########
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
def generate_forecast_plot(data):
    # Plot the original time series data
    st.subheader("Original Time Series Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='lines', name='Actual'))
    fig.update_layout(title='Original Time Series Data', xaxis_title='Date', yaxis_title='Amount')
    st.plotly_chart(fig)

    order = (0,0,6)
    seasonal_order = (0, 0, 2, 12)

    model = ARIMA(data['Amount'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Forecast for the next 12 months
    forecast_steps = 12
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate dates for the forecast period
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps + 1, freq='MS')[1:]

    # Plot the actual data, original amount values, and forecast
    st.subheader("Time-series Forecasting with ARIMA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='markers', name='Original Amount', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    fig.update_layout(title='Time-series Forecasting with ARIMA', xaxis_title='Date', yaxis_title='Amount')
    st.plotly_chart(fig)

    # Display forecasted values
    st.subheader("Forecasted Values:")
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Amount': forecast})
    st.write(forecast_df)
if st.button('Forecast'):
    generate_forecast_plot(data)

########
# Calculate error metrics
order = (0,0,6)
seasonal_order = (0, 0, 2, 12)
model = ARIMA(data['Amount'], order=order, seasonal_order=seasonal_order)
model_fit = model.fit()
forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)
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






