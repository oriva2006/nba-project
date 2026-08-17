import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

def load_data(file_path = "shots_cleaned.csv"):
    """
    Loads the cleaned shot data from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def prepare_features(df):
    """
    Prepares features and target variable for modeling.
    """
    # Define feature columns
    numerical_cols = [
        "PERIOD", 
        "MINUTES_REMAINING", 
        "SECONDS_REMAINING", 
        "SHOT_DISTANCE",
        "CALCULATED_DIST", 
        "SHOT_ANGLE",
        "SHOT_VALUE"
    ]
    categorical_cols = ["SHOT_ZONE_BASIC", "SHOT_TYPE", "ACTION_GROUP"]
    
    X = df[numerical_cols + categorical_cols]
    y = df["TARGET"]
    
    return X, y, numerical_cols, categorical_cols

def train_model(X, y, numerical_cols, categorical_cols):
    """
    Trains a machine learning model to predict shot success.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Define the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier())
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    
    print("✅ Model Evaluation Metrics:")
    print(f"Log Loss: {logloss:.4f}(lower is better)")
    print(f"ROC AUC: {roc_auc:.4f} (0.5 is baseline 1.0 is perfect)")
    print(f"Brier Score: {brier_score:.4f} (mean squared error of probabilistic predictions, lower is better)")

    
    return model_pipeline

def save_model(model, file_path = "shot_success_model.pkl"):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, file_path)
    print(f"✅ Model saved to {file_path}")

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    X, y, numerical_cols, categorical_cols = prepare_features(df)
    
    # Train the model
    model = train_model(X, y, numerical_cols, categorical_cols)
    
    # Save the trained model
    save_model(model)