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
