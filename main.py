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
        y = df['Sales']
        # Drop identifier and target columns from features
        X = df.drop(['Sales'], axis=1, errors='ignore')
     
        enc = OrdinalEncoder()
        encoded = enc.fit_transform(y)
        print(encoded)

        # Create and show a plot
        if 'type' in df.columns:
            fig = px.pie(df, names='type', title='Distribution of Transaction Types')
            fig.show()
        else:
            print("No 'type' column found in the dataset")
        
       
        # Align X and y and split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2
        )
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        print("X_test:", X_test[:100])
        print("Test score:", clf.score(X_test, y_test))
        print("Sample predictions:", clf.predict(X_test)[:100])
    else:
        print("Could not proceed without data.")



if __name__ == "__main__":
    main()
