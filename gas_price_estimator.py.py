import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Nat_Gas.csv")
df.columns = ['Date', 'Price']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Create time index
df['t'] = np.arange(len(df))

# Extract month
df['month'] = df['Date'].dt.month

# Create seasonal dummies
month_dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True)

# Prepare features
X = pd.concat([df[['t']], month_dummies], axis=1)
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Pricing function
def estimate_price(input_date):
    input_date = pd.to_datetime(input_date)
    
    # Time index calculation
    t = (input_date.year - df['Date'].dt.year.iloc[0]) * 12 + \
        (input_date.month - df['Date'].dt.month.iloc[0])
    
    # Seasonal encoding
    month = input_date.month
    dummy = np.zeros(len(month_dummies.columns))
    
    if month > 1:
        dummy[month - 2] = 1
    
    # Prediction
    X_input = np.concatenate([[t], dummy]).reshape(1, -1)
    return float(model.predict(X_input)[0])

# Example usage
print(estimate_price("2025-03-31"))
print(estimate_price("2025-12-31"))


