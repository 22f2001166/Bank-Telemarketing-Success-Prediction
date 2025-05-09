import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train = pd.read_csv("/kaggle/input/predict-the-success-of-bank-telemarketing/train.csv")
test = pd.read_csv("/kaggle/input/predict-the-success-of-bank-telemarketing/test.csv")


# Feature engineering
def create_features(df):
    # Create interaction features
    df["age_campaign"] = df["age"] * df["campaign"]
    df["balance_duration"] = df["balance"] * df["duration"]

    # Create categorical feature combinations
    df["job_education"] = df["job"] + "_" + df["education"]

    # Binning numerical features
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Middle", "Senior", "Elder"],
    )
    df["balance_group"] = pd.cut(
        df["balance"],
        bins=[-1, 0, 1000, 5000, 100000],
        labels=["No Balance", "Low", "Medium", "High"],
    )

    return df


# Apply feature engineering
train = create_features(train)
test = create_features(test)

# Update feature lists
categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "poutcome",
    "job_education",
    "age_group",
    "balance_group",
]
numerical_features = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "age_campaign",
    "balance_duration",
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Prepare data
X = train.drop("target", axis=1)
y = train["target"].map({"yes": 1, "no": 0})


# Multiple resampling techniques
def get_resampled_data(X_processed, y, method="smote"):
    if method == "smote":
        resampler = SMOTETomek(random_state=42)
    elif method == "adasyn":
        resampler = ADASYN(random_state=42)

    return resampler.fit_resample(X_processed, y)


# Preprocess data
X_processed = preprocessor.fit_transform(X)
# Try different resampling methods
X_res, y_res = get_resampled_data(X_processed, y, method="adasyn")

# Stratified K-Fold for more robust validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Comprehensive hyperparameter grid
param_grid = {
    "rf": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    },
    "gb": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.6, 0.8, 1.0],
    },
    "svm": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01],
        "kernel": ["rbf", "poly"],
    },
}

# Base models with improved parameters
base_models = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
]

# Stacking Classifier for improved ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(solver="liblinear", random_state=42),
    cv=5,
)


# Hyperparameter tuning function
def tune_model(model, param_grid, X, y):
    grid_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1_macro",
        cv=skf,
        random_state=42,
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_


# Tune individual models
tuned_models = []
for name, model in base_models:
    if name in param_grid:
        tuned_model = tune_model(model, param_grid[name], X_res, y_res)
        tuned_models.append((name, tuned_model))


# Fit stacking classifier
stacking_clf.fit(X_res, y_res)


# Validation
def validate_model(model, X, y):
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scores.append(
            {
                "f1_score": f1_score(y_val, y_pred),
                "roc_auc": roc_auc_score(y_val, y_pred),
            }
        )

    return pd.DataFrame(scores).mean()


# Evaluate models
print("Model Performance:")
print("Stacking Classifier:")
stacking_results = validate_model(stacking_clf, X_res, y_res)
print(stacking_results)


# Prepare test submission
test_processed = preprocessor.transform(test)
test_predictions = stacking_clf.predict(test_processed)
submission = pd.DataFrame(
    {"id": test.index, "target": np.where(test_predictions == 1, "yes", "no")}
)
submission.to_csv("submission.csv", index=False)


print("Submission file created!")
