import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os

# Load the saved pipeline
# Function to load the model pipeline
def load_pipeline(pipeline_path):
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except ModuleNotFoundError as e:
        st.error(f"ModuleNotFoundError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model pipeline: {e}")
        st.stop()

# Path to the model pipeline
pipeline_path = 'model/xgboost_model_pipeline.pkl'

# Load the saved pipeline
pipeline = load_pipeline(pipeline_path)

class PreprocessingPipeline:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        self.days = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
            'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        self.vehicle_prices = {
            'less than 20000': 1, '20000 to 29000': 2, '30000 to 39000': 3,
            '40000 to 59000': 4, '60000 to 69000': 5, 'more than 69000': 6,
        }
        self.age_of_vehicle_variants = {
            'new': 0.5, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8.5,
        }
        self.age_variants = {
            '16 to 17': 1, '18 to 20': 2, '21 to 25': 3, '26 to 30': 4,
            '31 to 35': 5, '36 to 40': 6, '41 to 50': 7, '51 to 65': 8, 'over 65': 9,
        }
        self.reverse_maps = {}

    def load_data(self, file_path):
        data = pd.read_excel(file_path)
        return data.copy()
    
    def process_months(self, df, month_columns):
        month_proc = lambda x: self.months.get(x, 0)
        for col in month_columns:
            df[col] = df[col].apply(month_proc)
        return df
    
    def process_days_of_week(self, df, day_columns):
        day_proc = lambda x: self.days.get(x, 0)
        for col in day_columns:
            df[col] = df[col].apply(day_proc)
        return df
    
    def process_vehicle_price(self, df, vehicle_price_column):
        vehicle_price_proc = lambda x: self.vehicle_prices.get(x, 0)
        df[vehicle_price_column] = df[vehicle_price_column].apply(vehicle_price_proc)
        return df
    
    def process_vehicle_age(self, df, vehicle_age_column):
        vehicle_age_proc = lambda x: self.age_of_vehicle_variants.get(x, 0)
        df[vehicle_age_column] = df[vehicle_age_column].apply(vehicle_age_proc)
        return df
    
    def process_policy_holder_age(self, df, age_column):
        age_proc = lambda x: self.age_variants.get(x, 0)
        df[age_column] = df[age_column].apply(age_proc)
        return df
    
    def fill_missing_values(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            df[column] = df[column].fillna(df[column].median())
        return df
    
    def handle_age(self, df, age_column):
        mean_age = df[age_column].mean()
        df[age_column] = df[age_column].apply(lambda x: mean_age if pd.isnull(x) or x < 16 else x)
        return df
    
    def object_to_numerical(self, df):
        for column in df.select_dtypes(include=['object']).columns:
            unique_values = df[column].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            reverse_map = {i: value for value, i in value_map.items()}
            self.reverse_maps[column] = reverse_map
            df[column] = df[column].map(value_map)
        return df
    
    def process_data(self, df):
        df = self.process_months(df, ['Month', 'MonthClaimed'])
        df = self.process_days_of_week(df, ['DayOfWeek', 'DayOfWeekClaimed'])
        df = self.process_vehicle_price(df, 'VehiclePrice')
        df = self.process_vehicle_age(df, 'AgeOfVehicle')
        df = self.process_policy_holder_age(df, 'AgeOfPolicyHolder')
        
        df = self.handle_age(df, 'Age')
        df = self.fill_missing_values(df)
        df = self.object_to_numerical(df)
        
        return df

def get_claim_details_by_policy_number(dataframe, policy_number):
    claim_details = dataframe[dataframe['PolicyNumber'] == policy_number]
    if claim_details.empty:
        raise ValueError(f"No claim found for policy number: {policy_number}")
    return claim_details

def predict_fraud(data, threshold=0.5):
    probabilities = pipeline.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities

def process_csv_for_inference(file_path):
    preprocessing_pipeline = PreprocessingPipeline()
    data = preprocessing_pipeline.load_data(file_path)
    
    # Remove duplicate PolicyNumbers
    data = data.drop_duplicates(subset=['PolicyNumber'])
    
    processed_data = preprocessing_pipeline.process_data(data)
    return processed_data, data, preprocessing_pipeline.reverse_maps

def process_policynumber_for_inference(file_path, policy_number):
    preprocessing_pipeline = PreprocessingPipeline()
    data = preprocessing_pipeline.load_data(file_path)
    
    # Remove duplicate PolicyNumbers
    data = data.drop_duplicates(subset=['PolicyNumber'])
    
    # Process data
    processed_data = preprocessing_pipeline.process_data(data)
    
    # Verify the policy number exists after preprocessing
    if policy_number not in processed_data['PolicyNumber'].values:
        raise ValueError(f"No claim found for policy number: {policy_number} after preprocessing")
    
    claim_details = processed_data[processed_data['PolicyNumber'] == policy_number]
    original_claim_details = data[data['PolicyNumber'] == policy_number]
    
    if claim_details.empty:
        raise ValueError(f"No claim found for policy number: {policy_number} after processing details")
    
    return processed_data, claim_details, original_claim_details, preprocessing_pipeline.reverse_maps

def predict_from_csv(file_path, threshold=0.5):
    processed_data, original_data, reverse_maps = process_csv_for_inference(file_path)
    predictions, probabilities = predict_fraud(processed_data, threshold)
    
    original_data['FraudProbability'] = probabilities
    original_data['FraudProbability'] = original_data['FraudProbability'].round(2)
    original_data['Prediction'] = predictions
    original_data['Prediction'] = original_data['Prediction'].map({0: 'Legit', 1: 'Possible Fraud'})
    original_data = original_data.drop_duplicates(subset=['PolicyNumber'])
    
    # Reverting numerical values back to original categorical values
    for column, reverse_map in reverse_maps.items():
        original_data[column] = original_data[column].map(reverse_map)
    
    original_data['Age'] = original_data['Age'].astype(int)   # Round age to the nearest number
    
    return original_data[['PolicyNumber', 'Make', 'Sex', 'Age', 'FraudProbability', 'Prediction']]

def predict_from_policy_number(file_path, policy_number, threshold=0.5):
    processed_data, claim_details, original_data, reverse_maps = process_policynumber_for_inference(file_path, policy_number)
    predictions, probabilities = predict_fraud(claim_details, threshold)
    # Add prediction results to the processed claim details
    claim_details['FraudProbability'] = probabilities
    claim_details['Prediction'] = predictions
    claim_details['Prediction'] = claim_details['Prediction'].map({0: 'Legit', 1: 'Possible Fraud'})
    claim_details['Age'] = claim_details['Age'].astype(int)  # Ensure age is a whole number
    
    # Drop duplicates if any
    claim_details = claim_details.drop_duplicates(subset=['PolicyNumber'])
    
    # Reverting numerical values back to original categorical values
    for column, reverse_map in reverse_maps.items():
        claim_details[column] = claim_details[column].map(reverse_map).fillna(claim_details[column])
    
    return claim_details[['PolicyNumber', 'Make', 'Sex', 'Age', 'FraudProbability', 'Prediction']]

def format_predictions(df):
    df['FraudProbability'] = df['FraudProbability'].apply(lambda x: f"<b>{x}</b>")
    df['Prediction'] = df.apply(lambda row: f"<b><span style='color: red;'>{row['Prediction']}</span></b>" 
                                     if row['Prediction'] == 'Possible Fraud' else f"<b>{row['Prediction']}</b>", axis=1)
    return df

def main():
    st.set_page_config(page_title="AI Fraud Detection", layout="wide")
    
    # Background setup
    def set_background(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{encoded_string});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    # Set the custom page title with color
    st.markdown("""
        <style>
        .title {
            color: #FCB115;  /* Change this to your desired color */
            font-size: 36px;
            font-weight: bold;
        }
        </style>
        <div class="title">Intelligent Auto Insurance Fraud Detection System</div>
    """, unsafe_allow_html=True)
    
    set_background('data/AI-Fraud-Detection.jpg')  # Update with your correct path

    st.sidebar.title("About")
    about_expander = st.sidebar.expander("About", expanded=False)
    with about_expander:
        st.write("""
            Welcome to the Intelligent Auto Insurance Fraud Detection System. This application leverages artificial intelligence to identify potential fraudulent insurance claims with high accuracy. 
            Users can upload their own claim datasets for analysis, or the system can utilize a default dataset for predictions if no file is provided(For demos). Enter or select a policy number for individual claim evaluation, or process an entire dataset for batch fraud detection. 
            Adjust the fraud probability threshold to fine-tune the sensitivity of the predictions and download the results for further review.
            
        """)
    
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a claim file in Excel", type=["xlsx"])
    threshold = st.sidebar.slider("Set Fraud Probability Threshold", 0.0, 1.0, 0.5)
    
    if uploaded_file is not None:
        # Load policy numbers from the uploaded file
        data = pd.read_excel(uploaded_file)
        policy_numbers = data['PolicyNumber'].astype(str).unique().tolist()
    else:
        # Load policy numbers from the default file
        file_path = "data/default-2-NewNew.xlsx"
        data = pd.read_excel(file_path)
        policy_numbers = data['PolicyNumber'].astype(str).unique().tolist()
   
    # Create dropdown for policy numbers
    policy_number_suggestions = st.sidebar.selectbox(
        "Select or enter a Policy Number",
        options=[""] + policy_numbers,
        key="policy_number_suggestions"
    )
    policy_number = policy_number_suggestions

    # Process data
    if st.sidebar.button("Submit"):
        try:
            with st.spinner("Processing..."):
                if uploaded_file:
                    if policy_number:
                        result = predict_from_policy_number(uploaded_file, int(policy_number), threshold)
                    else:
                        result = predict_from_csv(uploaded_file, threshold)
                else:
                    if policy_number:
                        result = predict_from_policy_number(file_path, int(policy_number), threshold)
                    else:
                        result = predict_from_csv(file_path, threshold)
                
                result = format_predictions(result)
            
            # Custom CSS to style the table
            st.write("""
                <style>
                table {
                    width: 100%;
                    background-color: rgba(255, 255, 255, 0.6);
                    border: 1px solid black;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                    font-size: 14px;
                    font-weight: bold;
                    color: black;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.write(result.to_html(escape=False), unsafe_allow_html=True)

            # Move download button to sidebar
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(result)
            st.sidebar.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
