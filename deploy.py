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

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
  # Replace 'data.csv' with the actual path to your data file

# Get the list of categorical variables
categorical_variables = ['Gender', 'Age_Group', 'Status', 'Category']

# Function to create a bar chart for a categorical variable
def create_bar_chart(var):
    chart_data = data[var].value_counts()
    st.bar_chart(chart_data)

# Function to create a pie chart for a categorical variable
def create_pie_chart(var):
    fig, ax = plt.subplots(figsize=(5, 5))
    data[var].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title(f'Percentage Distribution of {var}')
    return fig

# Function to create a grouped bar chart for a categorical variable
def create_grouped_bar_chart(var):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x=var, hue='Age_Group', ax=ax)
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.set_title(f'Grouped bar chart of {var} by Age Group')
    tick_positions = [i for i, _ in enumerate(data[var].unique())]
    ax.set_xticks(tick_positions, labels=data[var].unique(), rotation=45)
    ax.legend(title='Age Group')
    return fig

# Layout setup
st.title("Data Analysis")

# Create a row layout for dropdown menus
col1, col2, col3 = st.columns(3)

# Dropdown menu for selecting categorical variables
with col1:
    selected_var1 = st.selectbox("Frequency Distribution:", categorical_variables)
    #st.title(f"Frequency Distribution of {selected_var1}")
    create_bar_chart(selected_var1)

with col2:
    selected_var2 = st.selectbox("Percentage Distribution:", categorical_variables)
    #st.title(f'Percentage Distribution of {selected_var2}')
    fig2 = create_pie_chart(selected_var2)
    st.pyplot(fig2)

with col3:
    selected_var3 = st.selectbox("Grouped Bar Chart:", categorical_variables)
    #st.title(f'Grouped Bar Chart of {selected_var3} by Age Group')
    fig3 = create_grouped_bar_chart(selected_var3)
    st.pyplot(fig3)



###################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Function to plot the histogram for top states
def plot_top_states_histogram(num_states):
    # Get the top states with the highest number of orders
    top_states = data['ship_state'].value_counts().nlargest(num_states)

    # Plotting the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    top_states.plot(kind='bar', ax=ax)

    # Annotate each bar with its order count
    for i, count in enumerate(top_states):
        ax.text(i, count + 1, str(count), ha='center', va='bottom')

    ax.set_title(f'Top {num_states} States with Highest Number of Orders')
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Orders')
    ax.set_xticklabels(top_states.index, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit UI
st.title('Top States with Highest Number of Orders')

# Slider to select the number of states
num_states = st.slider('Select Number of States', min_value=1, max_value=50, value=10, step=1)

# Plot the histogram for selected number of states
plot_top_states_histogram(num_states)
























import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame called 'data'
# Replace 'data.csv' with the actual path to your data file if it's in a CSV format


# Group the data by 'ship-state' and 'ship-city', then count the orders for each city within each state
state_city_orders = data.groupby(['ship_state', 'ship-city']).size().reset_index(name='order_count')

# Function to get top 5 cities with maximum order count for each state
def top_5_cities(df):
    return df.nlargest(5, 'order_count')

# Apply the function to get top 5 cities for each state
top_cities_by_state = state_city_orders.groupby('ship_state').apply(top_5_cities)

# Reset index to make 'ship-state' and 'ship-city' regular columns
top_cities_by_state.reset_index(drop=True, inplace=True)

# Streamlit app
st.title('Top 5 Cities with Highest Number of Orders')

# Dropdown menu for selecting ship state
selected_state = st.selectbox('Select a Ship State:', top_cities_by_state['ship_state'].unique())

# Get top cities and order counts for the selected state
cities = top_cities_by_state[top_cities_by_state['ship_state'] == selected_state]['ship-city']
order_counts = top_cities_by_state[top_cities_by_state['ship_state'] == selected_state]['order_count']

# Plotting the histogram for ship cities
plt.figure(figsize=(10, 6))
plt.bar(cities, order_counts)
plt.title('Top 5 Cities with Highest Number of Orders in {}'.format(selected_state))
plt.xlabel('Ship City')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot using Streamlit
st.pyplot(plt)



############


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









