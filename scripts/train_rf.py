import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

print("Loading dataset...")
df = pd.read_csv(r'd:\SLIIT 4 Y 2S\ML\Project\ML-assignment\data\energy_dataset.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month

features = ['total load actual', 'generation solar', 'generation wind onshore', 'hour', 'month']
target = 'price actual'

df_clean = df[features + [target]].dropna()
X = df_clean[features]
y = df_clean[target]

print(f"Dataset size: {len(df_clean)} rows")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest (10 estimators)...")
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Results for Random Forest:")
print(f"MAE: {mae:.4f} EUR/MWh")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)
