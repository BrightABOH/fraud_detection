# Overview

The Intelligent Auto Insurance Fraud Detection System is a powerful AI-driven application designed to identify potential fraudulent insurance claims. By leveraging machine learning techniques and a robust preprocessing pipeline, this system provides accurate predictions to help mitigate fraud risks. Users can upload their own claim datasets or rely on a default dataset for fraud detection.

# Features
- AI-Powered Fraud Detection: Utilizes advanced machine learning models to predict the likelihood of fraud in insurance claims.
- Flexible Data Input: Accepts user-uploaded datasets or uses a built-in default dataset.
- Policy Number Lookup: Allows for single claim analysis by entering or selecting a policy number.
- Batch Processing: Processes entire datasets for comprehensive fraud detection.
- Adjustable Threshold: Fine-tune the fraud probability threshold to control sensitivity.
- Interactive Visualization: Presents results in an easily interpretable format with options to download predictions.

# Default Dataset
 If a user does not upload a claim dataset, the application will automatically use a pre-configured default dataset. This default dataset contains a representative sample of insurance claims, ensuring that fraud predictions can still be generated even without user-provided data. The default dataset is useful for:

- Testing and Demonstration: Allows users to explore the functionality and capabilities of the application without needing their own data.
- Benchmarking: Provides a baseline for evaluating the performance of the fraud detection system.
- Backup Analysis: Ensures that fraud detection predictions can be made even if no user data is uploaded.

# How It Works
- Default Dataset Handling: If no file is uploaded, the application uses a default dataset stored in the project directory.
- Processing and Prediction: The default dataset is processed using the same preprocessing pipeline as user-uploaded data, and predictions are made using the trained machine learning model.
- Result Display: Results from the default dataset are displayed in the application interface, and users can view and download these predictions.


# Installation
- Clone the Repository
```sh
git clone https://github.com/BrightABOH/fraud-detection.git
cd fraud-detection
```
- Install Dependencies
```sh
pip install -r requirements.txt
```

# Usage
1. Run the Streamlit Application
```sh
streamlit run stream_app.py

```
2. Navigate to the Application
- Open your web browser and go to http://localhost:8501
3. Use the Application
- Upload a Claim File: Use the sidebar to upload an Excel file containing insurance claim data.
- Set Fraud Probability Threshold: Adjust the slider to set the threshold for fraud prediction.
- Enter or Select a Policy Number: Enter a policy number manually or select from the dropdown list.
- Submit for Analysis: Click the "Submit" button to process the data and view results.
If no file is uploaded, the system will automatically use the default dataset for analysis.

# Application Interface

- About: Provides information about the application.
- Upload File: Allows users to upload their claim datasets.
- Threshold Slider: Adjusts the sensitivity of fraud detection.
- Policy Number Input: For single claim analysis.
- Results Display: Shows predictions and probabilities for each claim.
- Download Button: Exports the results to a CSV file.

# Background Image

To set a custom background image for the application, place your image file in the specified directory and update the path in the set_background function.

# Contribution

We welcome contributions to improve this project! Feel free to open issues or submit pull requests.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Contact
For any inquiries or support, please contact:

Bright Aboh
Website: https://brightaboh.github.io
LinkedIn: https://www.linkedin.com/in/bright-aboh-b85932ba/
Email: bright.s.e.aboh@aims-senegal.org
