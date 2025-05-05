import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Define features and target
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Year']
X = df[features]
y = df['Weekly_Sales']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to rf_model.pkl")
