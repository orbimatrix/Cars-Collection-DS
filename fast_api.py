from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn

app = FastAPI(title="Car Price Prediction API", version="1.0")

try:
    model = joblib.load('models/car_price_model.pkl')
    pt = joblib.load('models/power_transformer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    pca = joblib.load('models/pca.pkl')
    train_cols = joblib.load('models/train_columns.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")

# Define the Input Schema using Pydantic
class CarInput(BaseModel):
    company: str
    fuel_type: str
    hp: float
    cc: float
    torque: float
    speed: float
    acceleration: float
    seats: int

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is Online", "docs": "/docs"}

@app.post("/predict")
def predict_price(data: CarInput):
    try:
        # 1. Numeric Transformation (Yeo-Johnson)
        # PowerTransformer expects: [Price_Dummy, Torque, HP, CC]
        dummy_price = 0
        num_arr = np.array([[dummy_price, data.torque, data.hp, data.cc]])
        num_fixed = pt.transform(num_arr)
        
        torque_f, hp_f, cc_f = num_fixed[0, 1], num_fixed[0, 2], num_fixed[0, 3]

        # 2. Scaling
        scaled_vals = scaler.transform([[data.speed, data.acceleration]])
        speed_s, acc_s = scaled_vals[0]

        # 3. Encoding
        encoded_cat = encoder.transform([[data.company, data.fuel_type]])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['Company', 'Fuel_Type']))

        # 4. Assemble & Reindex
        input_df = pd.DataFrame([[hp_f, cc_f, torque_f, speed_s, acc_s, data.seats]], 
                                columns=['HP_Fixed', 'CC_Fixed', 'Torque_Fixed', 'Speed_Scaled', 'Acc_Scaled', 'Seats'])
        
        current_input = pd.concat([input_df, encoded_df], axis=1)
        
        # 5. PCA Alignment
        features_for_pca = [col for col in train_cols if col not in ['PCA1', 'PCA2', 'Target_Price_Fixed', 'Is_Anomaly']]
        pca_input = current_input.reindex(columns=features_for_pca, fill_value=0)
        pca_vals = pca.transform(pca_input)
        
        current_input['PCA1'] = pca_vals[:, 0]
        current_input['PCA2'] = pca_vals[:, 1]

        # 6. Final alignment for Model
        final_X = current_input.reindex(columns=[c for c in train_cols if c not in ['Target_Price_Fixed', 'Is_Anomaly']], fill_value=0)

        # 7. Predict & Inverse Transform
        pred_fixed = model.predict(final_X)
        
        price_dummy = np.zeros((1, 4))
        price_dummy[0, 0] = pred_fixed[0]
        final_usd = pt.inverse_transform(price_dummy)[0, 0]

        return {
            "status": "success",
            "predicted_price_usd": round(float(final_usd), 2),
            "currency": "USD"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)