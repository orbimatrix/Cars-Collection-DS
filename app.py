import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load all artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('models/car_price_model.pkl')
    pt = joblib.load('models/power_transformer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    pca = joblib.load('models/pca.pkl')
    encoder = joblib.load('models/encoder.pkl')

    # We also need the original columns used during training
    train_cols = joblib.load('models/train_columns.pkl') 
    return model, pt, scaler, encoder, train_cols,pca

model, pt, scaler, encoder, train_cols,pca = load_assets()

st.title("🚗 Pro Car Price Predictor")
st.markdown("Enter the vehicle specifications below to estimate the market value.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", encoder.categories_[0])
    fuel = st.selectbox("Fuel Type", encoder.categories_[1])
    hp = st.number_input("Horsepower (HP)", min_value=50, max_value=2000, value=150)
    cc = st.number_input("Engine Capacity (CC)", min_value=500, max_value=10000, value=2000)

with col2:
    speed = st.number_input("Top Speed (km/h)", min_value=100, max_value=500, value=200)
    accel = st.number_input("0-100 km/h (sec)", min_value=1.5, max_value=30.0, value=8.0)
    seats = st.slider("Seats", 2, 8, 5)
    torque = st.number_input("Torque (Nm)", min_value=50, max_value=2000, value=250)

if st.button("Estimate Price"):
    # 1. Transform numeric inputs (PowerTransformer expects 4 cols: Price, Torque, HP, CC)
    dummy_array = np.array([[0, torque, hp, cc]])
    numeric_fixed = pt.transform(dummy_array)
    torque_fixed, hp_fixed, cc_fixed = numeric_fixed[0, 1], numeric_fixed[0, 2], numeric_fixed[0, 3]
    
    # 2. Scale Speed and Accel
    speed_scaled, acc_scaled = scaler.transform([[speed, accel]])[0]
    
    # 3. Encode Categories
    encoded_cat = encoder.transform([[company, fuel]])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['Company', 'Fuel_Type']))
    
    # 4. Assemble the initial DataFrame
    input_df = pd.DataFrame([[hp_fixed, cc_fixed, torque_fixed, speed_scaled, acc_scaled, seats]], 
                            columns=['HP_Fixed', 'CC_Fixed', 'Torque_Fixed', 'Speed_Scaled', 'Acc_Scaled', 'Seats'])
    
    # Combine with One-Hot encoded columns
    current_input = pd.concat([input_df, encoded_df], axis=1)
    
    features_for_pca = [col for col in train_cols if col not in ['PCA1', 'PCA2', 'Target_Price_Fixed', 'Is_Anomaly']]
    
    pca_input = current_input.reindex(columns=features_for_pca, fill_value=0)
    
    # 6. Generate PCA Features
    pca_vals = pca.transform(pca_input)
    current_input['PCA1'] = pca_vals[:, 0]
    current_input['PCA2'] = pca_vals[:, 1]
    
    # 7. Final Reindex for the Model (Includes PCA columns now)
    final_model_input = current_input.reindex(columns=[c for c in train_cols if c not in ['Target_Price_Fixed', 'Is_Anomaly']], fill_value=0)
    
    # 8. Predict
    prediction_fixed = model.predict(final_model_input)
        
    # 9. Inverse Transform to USD
    price_dummy = np.zeros((1, 4))
    price_dummy[0, 0] = prediction_fixed[0]
    final_usd = pt.inverse_transform(price_dummy)[0, 0]
    
    st.success(f"### Estimated Price: ${final_usd:,.2f}")