# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Housing.csv")

# Input features
X = data[['square_feet', 'bedrooms', 'bathrooms']]

# Output target
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict prices
predictions = model.predict(X_test)

# Show predictions
print("Predicted Prices:")
print(predictions)

# Accuracy score
score = model.score(X_test, y_test)
print("\\nModel Accuracy:", score)

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()