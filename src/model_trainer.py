import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_car_model(data_path):
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Target_Price_Fixed', 'Is_Anomaly'])
    y = df['Target_Price_Fixed']
    
    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_cols = X_train.columns.tolist()
    joblib.dump(train_cols, 'models/train_columns.pkl')
    
    # Train the Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)
    
    # Calculate Metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Training loss (MSE on training set)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    
    print(f"✅ Model Training Success!")
    print(f"\n📊 Test Set Metrics:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE (Mean Absolute Error): ${mae:,.2f}")
    print(f"   MSE (Mean Squared Error): {mse:,.2f}")
    print(f"   RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    
    print(f"\n📈 Training Metrics:")
    print(f"   Training RMSE: ${train_rmse:,.2f}")
    print(f"   Training MSE: {train_mse:,.2f}")
    
    print(f"\n🔄 Cross-Validation Scores (5-fold):")
    print(f"   Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n⭐ Top 5 Important Features:")
    print(feature_importance.head())
    
    # Create visualizations
    _create_evaluation_plots(y_test, predictions, y_train, train_predictions, feature_importance)
    
    # Save the model artifact
    joblib.dump(model, 'models/car_price_model.pkl')
    return model

def _create_evaluation_plots(y_test, test_pred, y_train, train_pred, feature_importance):
    """Create evaluation plots for the model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation Plots', fontsize=16, fontweight='bold')
    
    # Plot 1: Predictions vs Actual (Test Set)
    axes[0, 0].scatter(y_test, test_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price')
    axes[0, 0].set_ylabel('Predicted Price')
    axes[0, 0].set_title('Test Set: Predictions vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals (Test Set)
    residuals = y_test - test_pred
    axes[0, 1].scatter(test_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot (Test Set)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top 10 Feature Importance
    top_features = feature_importance.head(10)
    axes[1, 0].barh(top_features['Feature'], top_features['Importance'], color='purple')
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_title('Top 10 Feature Importance')
    axes[1, 0].invert_yaxis()
    
    # Plot 4: Loss comparison (MSE)
    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse = np.mean((y_test - test_pred) ** 2)
    
    sets = ['Training Set', 'Test Set']
    mse_values = [train_mse, test_mse]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = axes[1, 1].bar(sets, mse_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Mean Squared Error (MSE)')
    axes[1, 1].set_title('Loss Comparison (Lower is Better)')
    
    # Add value labels on bars
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mse:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/evaluation_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n📉 Evaluation plots saved to 'models/evaluation_plots.png'")
    plt.close()

if __name__ == "__main__":
    train_car_model('data/final_feature_set.csv')


# ⭐ Top 5 Important Features:
#          Feature  Importance
# 2       HP_Fixed    0.693338
# 4   Torque_Fixed    0.145777
# 64          PCA1    0.072216
# 3       CC_Fixed    0.017409
# 65          PCA2    0.012855

#  We can see the PCA1 and pCA2 contribute most ( car brand and it's fuel)