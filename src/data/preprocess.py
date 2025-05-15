import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load ECG data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(file_path)

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the ECG data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column is the target
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/mitbih_train.csv"
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}") 