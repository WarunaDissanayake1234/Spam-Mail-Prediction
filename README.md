# Spam Mail Prediction with Logistic Regression

This repository contains code for a machine learning model that predicts whether an email is spam or ham (non-spam) using Logistic Regression.

## Overview

This project implements a Logistic Regression model to predict whether an email is spam or ham. Python programming language is used for implementation, and Google Colab served as the development environment. The features are extracted using `sklearn.feature_extraction.text`.

## Requirements

- Python
- Jupyter Notebook or Google Colab (for running the code)
- Libraries: pandas, scikit-learn

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/WarunaDissanayake1234/Spam-Mail-Prediction.git
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas scikit-learn
   ```

3. **Download the Dataset:**
   - Download the Spam ham dataset from [Kaggle](https://www.kaggle.com/datasets/bagavathypriya/spam-ham-dataset).
   - Place the dataset in the `data/` directory.

4. **Run the Code:**
   - Open the Jupyter Notebook or upload the provided notebook to Google Colab.
   - Follow the instructions within the notebook to preprocess the data, implement Logistic Regression, and predict whether an email is spam or ham.

## Files

- `spam_mail_prediction.ipynb`: Jupyter Notebook containing the code for spam mail prediction using Logistic Regression.
- `data/`: Directory containing the Spam ham dataset obtained from [Kaggle](https://www.kaggle.com/datasets/bagavathypriya/spam-ham-dataset).

## Dataset

The dataset used in this project includes emails labeled as spam or ham. The Logistic Regression model aims to predict the classification of emails based on their text content.

## Acknowledgments

The Spam ham dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/bagavathypriya/spam-ham-dataset).
