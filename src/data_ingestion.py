import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    return pd.read_csv(file_path)

if __name__ == "__main__":
    df=pd.read_csv('data\Cars Datasets 2025.csv',encoding='latin1')

    df.to_csv('data/raw_data.csv', index=False)
    print("Data Ingested successfully.")