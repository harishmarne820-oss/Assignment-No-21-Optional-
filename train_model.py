import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("insurance.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Separate Features and Target
X = df.drop("expenses", axis=1)
y = df["expenses"]

# One-Hot Encoding
X = pd.get_dummies(X)

# Save Encoded Column Names
encoded_columns = X.columns
joblib.dump(encoded_columns, "columns.pkl")

# Numerical Columns
numerical_cols = ["age", "bmi", "children"]

# Feature Scaling
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "LR_insurance.pkl")

# Prediction
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = r2_score(y_test, y_pred)

print("Model Trained Successfully!")
print("Model Accuracy:", accuracy)
print("All .pkl files created successfully!")