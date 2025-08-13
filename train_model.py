import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Simulated training data (you can replace this with real data later)
X = pd.DataFrame({
    'avg_corr': [0.3, 0.6, 0.8, 0.2, 0.9],
    'hhi': [0.2, 0.4, 0.7, 0.3, 0.9],
    'volatility': [0.1, 0.2, 0.4, 0.15, 0.5]
})
y = ['Low', 'Medium', 'High', 'Low', 'High']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save the model
joblib.dump(model, "models/risk_model.pkl")
print("Model trained and saved to models/risk_model.pkl")
