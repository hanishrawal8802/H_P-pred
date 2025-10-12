import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Create a simple dataset
np.random.seed(42)
size = np.random.randint(1000, 3000, 100)
bedrooms = np.random.randint(1, 5, 100)
age = np.random.randint(1, 30, 100)

# Generate price: base_price + (size * 100) + (bedrooms * 50000) - (age * 2000) + noise
price = 50000 + (size * 100) + (bedrooms * 50000) - (age * 2000) + np.random.normal(0, 10000, 100)

# Create DataFrame
data = pd.DataFrame({
    'size_sqft': size,
    'bedrooms': bedrooms,
    'house_age': age,
    'price': price
})

print("Dataset sample:")
print(data.head())

# Prepare features and target
X = data[['size_sqft', 'bedrooms', 'house_age']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.2f}")

# Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
print("\nModel saved successfully!")

# Save the dataset for reference
data.to_csv('model/housing_data.csv', index=False)