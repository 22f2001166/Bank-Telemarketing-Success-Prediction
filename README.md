## Predicting the Success of Bank Telemarketing Campaigns

This project builds a machine learning pipeline to predict the success of bank telemarketing campaigns using customer and campaign-related data. It performs extensive feature engineering, applies resampling techniques to handle class imbalance, and uses stacked ensemble models for high-performance classification.

### Dataset 

The data is sourced from the Kaggle Competition and consists of:

train.csv: contains features and target labels

test.csv: contains features for prediction

### Features and Engineering

The script performs the following feature engineering tasks:

Interaction features: age_campaign, balance_duration

Categorical combinations: job_education

Binning: Converts continuous variables like age and balance into groups like age_group, balance_group

### Preprocessing

Data preprocessing is handled using ColumnTransformer with:

StandardScaler for numeric features

OneHotEncoder for categorical features

### Resampling Techniques

To combat class imbalance, the following are implemented:

- SMOTETomek

- ADASYN

### Models and Tuning

The project utilizes multiple models in an ensemble stacking strategy:

Random Forest

Gradient Boosting

SVM

Logistic Regression (as meta-classifier)

Hyperparameter tuning uses RandomizedSearchCV over a defined parameter grid with StratifiedKFold cross-validation.

### Evaluation Metrics

Models are evaluated using:

F1 Score

ROC AUC Score

Confusion Matrix

Classification Report


