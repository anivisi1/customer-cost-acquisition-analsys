import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

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
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return model, X, X_test, y_test, y_pred, mse, r2, feature_importance

# Function to visualize results and interpretation
def visualize_results(data, model, X, X_test, y_test, y_pred, mse, r2, feature_importance):
    st.title('Model Results and Interpretation')

    # Display model performance metrics
    st.subheader('Model Performance')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R^2 Score: {r2}')

    # Display coefficients
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    st.subheader('Model Coefficients')
    st.write(coefficients)

    # Example: Visualize predictions vs actual values
    st.subheader('Predictions vs Actual Values')
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(predictions.head(10))

    # Display feature importance
    st.subheader('Feature Importance')
    st.write(feature_importance)

    # Example: Visualize feature importance
    fig, ax = plt.subplots()
    sns.barplot(x='Feature', y='Importance', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Main function to integrate all parts and run Streamlit app
def main(data_file):
    st.title('Customer Acquisition Analysis')

    # Load data
    data = load_data(data_file)

    # Perform EDA
    perform_eda(data)

    # Perform feature engineering if needed
    data = feature_engineering(data)

    # Train and evaluate model
    model, X, X_test, y_test, y_pred, mse, r2, feature_importance = train_and_evaluate_model(data)

    # Visualize results and interpretation
    visualize_results(data, model, X, X_test, y_test, y_pred, mse, r2, feature_importance)

if __name__ == '__main__':
    data_file = 'customer_acquisition_data_with_features.csv'
    main(data_file)
