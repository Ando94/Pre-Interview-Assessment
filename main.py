import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Define the data path relative to the script location
DATA_PATH = Path(__file__).parent / 'data' / 'logatta.csv'

def load_data():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}")
        return None

def main():
    print("Pre-Interview Assessment...")
    # Load and check data
    df = load_data()

    if df is not None:
        print("Processing data...")
        print("\nFirst few rows:")
        print(df.head())

        # Define target and initial feature set
        # Target is the interview acceptance column in this dataset
        y = df['accepted for the interview']

        # Drop identifier and target columns from features
        X = df.drop(['accepted for the interview', 'EmployeeNumber'], axis=1)

        # Identify categorical columns to encode
        categorical_cols = ['BusinessTravel', 'MaritalStatus', 'OverTime', 'Gender']

        # Instantiate OrdinalEncoder correctly (no data passed to constructor)
        enc = OrdinalEncoder()

        # Encode only the categorical columns
        X[categorical_cols] = enc.fit_transform(X[categorical_cols])

        # Print a small sample of encoded features
        print(X.head())

        # Create and show a plot of the target distribution
        if 'accepted for the interview' in df.columns:
            fig = px.pie(df, names='accepted for the interview', title='Acceptance Distribution')
            fig.show()
        else:
            print("No 'accepted for the interview' column found in the dataset")

        # Convert target to binary (handles TRUE/FALSE strings)
        y = y.astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

        # Align X and y and split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        print("X_test:", X_test[:100])
        print("Test score:", clf.score(X_test, y_test))
        print("Sample predictions:", clf.predict(X_test)[:100])
    else:
        print("Could not proceed without data.")



if __name__ == "__main__":
    main()
