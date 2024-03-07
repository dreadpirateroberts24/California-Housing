import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load your dataset
df = pd.read_csv('housing.csv')  # Update this path

# Set up the main structure of the app
st.set_page_config(page_title="California Housing Analysis", layout="wide")

# Page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ['Introduction', 'Analysis', 'Linear Regression'])

if page == 'Introduction':
    st.title("California Housing Prices")
    st.image('image.jpeg', use_column_width=True)  # Update this path
    st.write("The data contains information from the 1990 California census. It has median housing prices for California districts.")
    st.dataframe(df.head())  # Display the first few rows of the dataset

elif page == 'Analysis':
    st.title("Data Analysis")

    # House Value Distribution by Location
    fig1 = px.scatter(df, x="longitude", y="latitude", color="median_house_value",
                      title="House Value Distribution by Location")
    st.plotly_chart(fig1, use_container_width=True)

    # Median House Value vs. Median Income
    fig2 = px.scatter(df,
                      x='median_income',
                      y='median_house_value',
                      title='Median House Value vs. Median Income',
                      labels={'median_income': 'Median Income', 'median_house_value': 'Median House Value'},
                      hover_data=['latitude', 'longitude'])  # These could be added if you wish to provide more context on hover.
    fig2.update_layout(transition_duration=500)

    # Display the second figure
    st.plotly_chart(fig2, use_container_width=True)
    # Calculate average median house value per rooms
    avg_house_value_per_rooms = df.groupby('total_rooms')['median_house_value'].mean().reset_index()

    # Create a bar chart
    fig3 = px.bar(avg_house_value_per_rooms,
                  x='total_rooms',
                  y='median_house_value',
                  title='Average Median House Value by Total Rooms',
                  labels={'median_house_value': 'Average Median House Value', 'total_rooms': 'Total Rooms'})

    fig3.update_layout(transition_duration=500)

    # Display the bar chart in Streamlit
    st.plotly_chart(fig3, use_container_width=True)

elif page == 'Linear Regression':

    st.title("Linear Regression Model")

    # Selecting only numeric features for modeling and excluding the target variable
    predictors = df.select_dtypes(include=[np.number]).drop(columns=['median_house_value'])

    # Target variable
    y = df['median_house_value']

    imputer = SimpleImputer(strategy='mean')

    # Creating a pipeline that first fills missing values then fits a model
    pipeline = make_pipeline(imputer, LinearRegression())

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=42)

    # Fitting the model on the training data
    pipeline.fit(X_train, y_train)

    # Making predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Displaying feature importance
    coefficients = model.coef_
    importance = np.abs(coefficients)

    # For multiple predictors, we visualize their importance
    fig_importance = px.bar(x=importance, y=predictors.columns, orientation='h',
                            labels={'x': 'Absolute Coefficient Value', 'y': ''},
                            title='Feature Importance (Linear Regression)')

    st.plotly_chart(fig_importance)


    # Plotting actual vs predicted values for median_house_value
    fig_regression = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                title='Actual vs. Predicted Median House Value', opacity=0.65)
    # Add a line for perfect predictions
    fig_regression.add_trace(px.line(x=y_test, y=y_test, labels={'x': 'Actual Values', 'y': 'Actual Values'}).data[0])
    fig_regression.update_traces(line=dict(dash='dash', color='red'), name='Ideal Prediction')

    st.plotly_chart(fig_regression, use_container_width=True)


    # Displaying regression metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R^2 Score: {r2}")
