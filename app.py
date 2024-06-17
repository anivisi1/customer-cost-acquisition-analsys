import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to load and preprocess data
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    return data

# Function for exploratory data analysis (EDA)
def perform_eda(data):
    st.title('Exploratory Data Analysis')

    # Display summary statistics
    st.subheader('Summary Statistics')
    st.write(data.describe())

    # Distribution of Marketing Spend
    st.subheader('Distribution of Marketing Spend')
    fig, ax = plt.subplots()
    sns.histplot(data['Marketing_Spend'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Marketing Spend')
    st.pyplot(fig)

    # Relationship between variables (example: Marketing Spend vs New Customers)
    st.subheader('Relationship: Marketing Spend vs New Customers')
    fig, ax = plt.subplots()
    sns.scatterplot(x='Marketing_Spend', y='New_Customers', data=data, ax=ax)
    ax.set_title('Marketing Spend vs New Customers')
    st.pyplot(fig)

# Function for feature engineering (if needed)
def feature_engineering(data):
    # Example: Convert categorical variables to dummy variables
    data = pd.get_dummies(data, columns=['Marketing_Channel'], drop_first=True)
    return data

# Function to train and evaluate the model
def train_and_evaluate_model(data):
    # Split data into features (X) and target variable (y)
    X = data.drop(['Customer_ID', 'New_Customers'], axis=1)
    y = data['New_Customers']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Function to make predictions
def make_prediction(model, X):
    # Perform prediction
    y_pred = model.predict(X)
    return y_pred

# Main function to integrate all parts and run Streamlit app
def main():
    st.title('Customer Acquisition Analysis')

    # Load data
    data_file = 'customer_acquisition_data_with_features.csv'
    data = load_data(data_file)

    # Perform EDA
    perform_eda(data)

    # Perform feature engineering if needed
    data = feature_engineering(data)

    # Train model and get necessary objects
    model, X_test, y_test = train_and_evaluate_model(data)

    # Display form for user input
    st.sidebar.title('Make a Prediction')
    st.sidebar.write('Enter values for prediction:')
    # Example: Add input fields for user to enter values
    input_features = {}
    for feature in X_test.columns:
        input_features[feature] = st.sidebar.number_input(f'Enter {feature}', min_value=0.0)

    if st.sidebar.button('Predict'):
        # Create a DataFrame from user input to match model's expected input format
        input_data = pd.DataFrame([input_features])
        # Perform prediction using the trained model
        prediction = make_prediction(model, input_data)

        # Display prediction
        st.sidebar.subheader('Prediction Results:')
        st.sidebar.write(f'Predicted New Customers: {prediction[0]:.2f}')

if __name__ == '__main__':
    main()
