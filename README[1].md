# AI Customer Support Ticket Triage

## Project Overview

An AI-based customer support system that automatically analyzes customer support tickets and predicts:

- Ticket Category
- Ticket Urgency
- Support Queue
- Prediction Confidence

The system also sends low-confidence predictions for human review.

## Categories

- Billing
- Technical
- Account
- Product

## Urgency Levels

- High
- Medium
- Low

## Technologies Used

- Python
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Pandas

## Machine Learning Pipeline

Customer Ticket
-> Text Cleaning
-> TF-IDF Vectorization
-> Category Prediction
-> Urgency Prediction
-> Confidence Check
-> Support Queue Routing

## Model Performance

Category model:
Approximately 90% accuracy on the current test split.

Urgency model:
Approximately 62.5% accuracy on the current test split.

These results are prototype results from a small synthetic dataset and should not be considered production performance.

## Features

- Automatic ticket classification
- Urgency detection
- Support-team routing
- Confidence-based human review
- Prediction history
- Dashboard statistics
- Google Drive model storage

## Future Enhancements

- Larger real-world dataset
- Transformer/embedding-based models
- Better urgency classification
- Cloud deployment
- Database integration
- Email/ticketing-system integration

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
