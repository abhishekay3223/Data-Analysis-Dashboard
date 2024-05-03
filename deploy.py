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
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Sales Dashboard",page_icon= "sales.png", layout="wide")
st.title(":bar_chart: Sales Dashboard")

#created a dataframe
data = pd.read_csv("Store Data.csv")

#Configuring the page
###########
def rectify_state_names(state_name):
  """
  This function corrects the state name in the `ship-state` column based on a provided list of states and union territories of India.

  Args:
      state_name (str): The state name to be corrected.

  Returns:
      str: The corrected state name (or the original name if no correction is needed).
  """

  # Define a dictionary to map incorrect names to their corrected versions
  state_corrections = {
      "PUNJAB": "PUNJAB",
      "HARYANA": "HARYANA",
      "WEST BENGAL": "WEST BENGAL",
      "TAMIL NADU": "TAMIL NADU",
      "MAHARASHTRA": "MAHARASHTRA",
      "KARNATAKA": "KARNATAKA",
      "ANDHRA PRADESH": "ANDHRA PRADESH",
      "KERALA": "KERALA",
      "ASSAM": "ASSAM",
      "TELANGANA": "TELANGANA",
      "DELHI": "DELHI",
      "ODISHA": "ODISHA",
      "RAJASTHAN": "RAJASTHAN",
      "UTTAR PRADESH": "UTTAR PRADESH",
      "MADHYA PRADESH": "MADHYA PRADESH",
      "UTTARAKHAND": "UTTARAKHAND",
      "ANDAMAN & NICOBAR ": "ANDAMAN AND NICOBAR ISLANDS",  # Handle extra space
      "GUJARAT": "GUJARAT",
      "CHANDIGARH": "CHANDIGARH",
      "JHARKHAND": "JHARKHAND",
      "BIHAR": "BIHAR",
      "HIMACHAL PRADESH": "HIMACHAL PRADESH",
      "PUDUCHERRY": "PUDUCHERRY",
      "DADRA AND NAGAR": "DADRA AND NAGAR HAVELI",
      "SIKKIM": "SIKKIM",
      "GOA": "GOA",
      "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
      "MANIPUR": "MANIPUR",
      "JAMMU & KASHMIR": "JAMMU AND KASHMIR",
      "TRIPURA": "TRIPURA",
      "New Delhi": "NEW DELHI",  # Handle variations of Delhi
      "CHHATTISGARH": "CHHATTISGARH",
      "LADAKH": "LADAKH",
      "MEGHALAYA": "MEGHALAYA",
      "NAGALAND": "NAGALAND",
      "MIZORAM": "MIZORAM",
      # Handle lowercase variations (assuming these are typos)
      "goa": "GOA",
      "punjab": "PUNJAB",
      "delhi": "NEW DELHI",
      "nagaland": "NAGALAND",
      "bihar": "BIHAR",
      # Handle extra spaces (assuming these are typos)
      "Arunachal pradesh": "ARUNACHAL PRADESH"
  }

  # Standardize the input state name (uppercase for consistency)
  state_name = state_name.upper()

  # Check if the state name exists in the correction dictionary
  if state_name in state_corrections:
    return state_corrections[state_name]
  else:
    # If no correction is found, return the original name
    return state_name
data['ship-city'] = data['ship-city'].str.upper()
# Example usage (assuming your data is in a DataFrame called 'data')
data['ship-state'] = data['ship-state'].apply(rectify_state_names)



###########





data.rename(columns={'Age Group': 'Age_Group', 'ship-state': 'ship_state'}, inplace=True)
#sidebar

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


df_selection = data.query(" Month == @month & ship_state == @shipstate")


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
import plotly.express as px



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

# Get the list of categorical variables
categorical_variables = ['Gender', 'Age_Group', 'Status', 'Category']

# Function to create a bar chart for a categorical variable using Plotly
def create_bar_chart(var):
    chart_data = data[var].value_counts().reset_index()
    chart_data.columns = [var, 'Count']
    fig = px.bar(chart_data, x=var, y='Count', labels={var: var, 'Count': 'Count'})
    st.plotly_chart(fig, height=300, width=300)

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
    fig.add_trace(go.Bar(x=top_states.index, y=top_states.values, text=top_states.values, textposition='auto'))

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
    fig.add_trace(go.Bar(x=top_cities.index, y=top_cities.values, text=top_cities.values, textposition='auto'))

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

    order = (0,0,6)
    seasonal_order = (0, 0, 2, 12)

    model = ARIMA(data['Amount'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Forecast for the next 12 months
    forecast_steps = 12
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate dates for the forecast period
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps + 1, freq='M')[1:]
    

# Sample data generation (replace this with your actual data)
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
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
    # st.subheader("Original Time Series Data")
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='lines', name='Actual'))
    # fig.update_layout(title='Original Time Series Data', xaxis_title='Date', yaxis_title='Amount')
    # st.plotly_chart(fig)

    order = (0,0,6)
    seasonal_order = (0, 0, 2, 12)

    model = ARIMA(data['Amount'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Forecast for the next 12 months
    forecast_steps = 12
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate dates for the forecast period
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_steps + 1, freq='M')[1:]

    # Plot the actual data, original amount values, and forecast
    amount_constant = 15000
    # st.subheader("Time-series Forecasting with ARIMA")
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='lines', name='Actual'))
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['Amount'], mode='markers', name='Original Amount', marker=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    # fig.update_layout(title='Time-series Forecasting with ARIMA', xaxis_title='Date', yaxis_title='Amount')
    # st.plotly_chart(fig)

    # Display forecasted values
    # st.subheader("Forecasted Values:")
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Amount': abs(forecast*amount_constant)})
    st.title('Forecasted Amount Viewer')
    selected_month = st.selectbox('Select Month ', forecast_df['Date'].dt.strftime('%B').unique())
    filtered_df = forecast_df[forecast_df['Date'].dt.strftime('%B') == selected_month]
    st.table(filtered_df[['Date', 'Forecasted Amount']].reset_index(drop=True))
generate_forecast_plot(data)










