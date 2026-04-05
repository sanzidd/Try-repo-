# 1. Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 2. Create a simple dataset
# X = input feature, y = output
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Convert to DataFrame (optional but common)
data = pd.DataFrame(X, columns=["Input"])
data["Output"] = y

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)

print("Predicted values:", y_pred)
print("Mean Squared Error:", mse)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)