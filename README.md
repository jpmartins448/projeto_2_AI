# Telecom Customer Churn Prediction Dashboard

## Project Overview
Telecom providers face revenue loss when customers cancel their subscriptions. This project delivers a churn prediction system that helps identify at-risk customers using a machine learning model and a Streamlit web app. The application predicts churn probability, assigns a risk level, and provides model insights to support retention strategies.

## Features
- Predict customer churn with probability and risk level.
- Interactive Streamlit dashboard with clean inputs and results.
- Model insights and evaluation plots for presentation and analysis.
- Model Training Playground to train and compare models inside the app.
- Customisable training parameters and hyperparameters.
- Save the best model as the production model.

## Machine Learning Pipeline
1. Data loading from a CSV dataset.
2. Feature selection and target separation (Churn).
3. Train/test split with stratification.
4. Optional feature scaling with StandardScaler.
5. Model training for multiple algorithms.
6. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
7. Model selection based on F1-score.
8. Saving the best pipeline and feature names for future predictions.

## Models Used
- Logistic Regression: A strong baseline linear model for classification.
- Decision Tree: A non-linear model that captures simple decision rules.
- Random Forest: An ensemble model that improves robustness and accuracy.

## Streamlit Web App
The dashboard is organised into tabs:
- Prediction: Enter customer details, predict churn, view probability and risk level.
- Model Insights: Visual plots for evaluation and understanding performance.
- Model Training Playground: Configure training settings, train models, compare results, and save the best model.

## How to Run the Project
Install dependencies and start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
- app.py: Streamlit web application.
- train_model.py: Training pipeline and evaluation script.
- customer_churn.csv: Dataset used for training and evaluation.
- churn_model.pkl: Saved production model pipeline.
- feature_names.pkl: Saved list of feature names used by the model.
- plots/: Folder containing evaluation and presentation plots.

## Example Usage
1. Enter customer information in the Prediction tab and click Predict Churn.
2. Review churn probability, risk level, and recommendation.
3. Open the Model Training Playground to select models and settings.
4. Train selected models, compare results, and review evaluation plots.
5. Save the best-performing model to replace the production model.

## Evaluation Metrics
- Accuracy: Overall proportion of correct predictions.
- Precision: How many predicted churns are correct.
- Recall: How many real churns are detected.
- F1-score: Balanced measure of precision and recall.
- Confusion matrix: Breakdown of correct and incorrect predictions.

F1-score is especially important for churn prediction because churned customers are often the minority class, and accuracy alone can be misleading.

## Future Improvements
- Train with real telecom data for stronger generalisation.
- Explore advanced models such as gradient boosting.
- Deploy the app to a cloud platform for wider access.
- Add explainable AI methods to justify predictions.
- Enable real-time scoring with streaming inputs.

## Disclaimer
This system is a proof of concept created for educational purposes. It is not intended for production use without further validation and real customer data.
