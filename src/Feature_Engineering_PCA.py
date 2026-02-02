import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def engineer_features(input_path):
    df = pd.read_csv(input_path, encoding='latin1')
    
    # 1. Encoding Categorical Data (One-Hot Encoding)
    # We turn 'Company' and 'Fuel_Type' into binary columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_cols = ['Company', 'Fuel_Type']
    encoded_data = encoder.fit_transform(df[cat_cols])
    
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=encoder.get_feature_names_out(cat_cols)
    )
    
    # Combine encoded categories with the numeric data
    # We use the Log-transformed versions for the final feature set
    numeric_features = ['CC_Scaled', 'HP_Scaled', 'Speed_Scaled', 'Seats', 'Torque_Log']
    final_df = pd.concat([df[numeric_features], encoded_df], axis=1)
    
    # 2. Anomaly Detection (Isolation Forest)
    # I used it because If the model tries to learn from a Ferrari SF90 (963 hp) as if it were a "normal" car, it will get confused and produce bad predictions for 99% of other cars which we will try to predict later on.
    iso = IsolationForest(contamination=0.05, random_state=42)
    final_df['Is_Anomaly'] = iso.fit_predict(final_df)
    
    # 3. PCA (Principal Component Analysis)
    # Reducing high-dimensional data (many companies/fuel types) into 2 components for 2D visuals
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(final_df.drop(columns=['Is_Anomaly']))
    final_df['PCA1'] = pca_results[:, 0]
    final_df['PCA2'] = pca_results[:, 1]
    
    final_df['Target_Price_Log'] = df['Price_Log']
    
    return final_df

if __name__ == "__main__":
    feature_set = engineer_features('data/preprocessed_cars.csv')
    feature_set.to_csv('data/final_feature_set.csv', index=False)
    print("Feature Engineering & PCA Complete.")
    print(f"Original features expanded to {feature_set.shape[1]} columns via Encoding.")