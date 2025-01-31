import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the dataset from the given file path."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column='math_score'):
    """
    Preprocess the data: handle missing values, encode categorical variables, and scale numerical features.
    If the target column is not present, it will skip dropping it.
    """
    # Separate features and target (if target column exists)
    if target_column in data.columns:
        X = data.drop(columns=[target_column])  # Drop the target column if it exists
        y = data[target_column]
    else:
        X = data  # If target column is not present, use all columns as features
        y = None  # No target column

    # Define categorical and numerical columns
    categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    numerical_cols = []

    # Preprocessing for numerical data (if any)
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

def split_data(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test