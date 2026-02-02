import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def clean_car_data(file_path):

    df = pd.read_csv(file_path, encoding='latin1')
    
    # 1. Helper function for numeric extraction values
    def extract_numeric(val):
        if pd.isna(val) or str(val).lower() == 'none' or str(val).strip() == '':
            return np.nan
        val = str(val).lower()
        # Remove currency symbols, commas, and unit strings
        val = re.sub(r'[\$,a-z/]', '', val)
        # Handle ranges (e.g., "12,000-15,000") by taking the mean
        if '-' in val:
            try:
                parts = [float(p.strip()) for p in val.split('-') if p.strip()]
                return sum(parts) / len(parts)
            except:
                return np.nan
        try:
            return float(val.strip())
        except:
            return np.nan

    # 2. Cleaning Phase
    processed_df = pd.DataFrame()
    processed_df['Company'] = df['Company Names'].str.strip()
    processed_df['Fuel_Type'] = df['Fuel Types'].str.strip()
    
    # Define columns to clean and their new names
    cols_map = {
        'CC/Battery Capacity': 'CC',
        'HorsePower': 'HP',
        'Total Speed': 'Speed',
        'Performance(0 - 100 )KM/H': 'Acceleration',
        'Cars Prices': 'Price',
        'Torque': 'Torque'
    }
    
    for raw_col, new_name in cols_map.items():
        processed_df[new_name] = df[raw_col].apply(extract_numeric)
    
    processed_df['Seats'] = pd.to_numeric(df['Seats'], errors='coerce')

    # 3. Imputation Phase 
    # Filling nulls based on similar car profiles like If a car is missing its "Torque" value, the imputer looks at 5 other cars with similar HP, CC, and Price to estimate the most logical Torque value.
    numeric_cols = ['CC', 'HP', 'Speed', 'Acceleration', 'Price', 'Torque', 'Seats']
    imputer = KNNImputer(n_neighbors=5)
    processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])

    # 4. Skewness Correction (Log  first)
    
    processed_df['Price_Log'] = np.log1p(processed_df['Price'])
    processed_df['Torque_Log'] = np.log1p(processed_df['Torque'])

    # 5. Scaling Phase
    # Scaling raw numeric features for models that are sensitive to magnitude (PCA, KNN, Linear)
    scaler = StandardScaler()
    scaled_features = ['CC', 'HP', 'Speed', 'Acceleration']
    processed_df[[f'{c}_Scaled' for c in scaled_features]] = scaler.fit_transform(processed_df[scaled_features])

    return processed_df

# Run the preprocessing
if __name__ == "__main__":
    cleaned_data = clean_car_data('data/Cars Datasets 2025.csv')
    cleaned_data.to_csv('data/preprocessed_cars.csv', index=False)
    print("Preproccesing Complete. File saved as 'preprocessed_cars.csv'")
    print(cleaned_data.head())