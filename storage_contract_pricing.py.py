import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==============================
# 1. LOAD YOUR DATA
# ==============================
df = pd.read_csv("natgas.csv")   # <-- make sure file name matches

df.columns = ['Date', 'Price']
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df = df.sort_values('Date')

# ==============================
# 2. BUILD PRICE MODEL
# ==============================
df['t'] = np.arange(len(df))
df['month'] = df['Date'].dt.month

# Seasonal dummies
month_dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True)

X = pd.concat([df[['t']], month_dummies], axis=1)
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# ==============================
# 3. PRICE ESTIMATION FUNCTION
# ==============================
def estimate_price(input_date):
    input_date = pd.to_datetime(input_date)

    t = (input_date.year - df['Date'].dt.year.iloc[0]) * 12 + \
        (input_date.month - df['Date'].dt.month.iloc[0])

    month = input_date.month
    dummy = np.zeros(len(month_dummies.columns))

    if month > 1:
        dummy[month - 2] = 1

    X_input = pd.DataFrame([np.concatenate([[t], dummy])], columns=X.columns)

    return float(model.predict(X_input)[0])

# ==============================
# 4. STORAGE CONTRACT PRICING
# ==============================
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    volume_per_trade,
    max_storage_volume,
    injection_rate,
    withdrawal_rate,
    storage_cost_per_month,
    injection_cost,
    withdrawal_cost
):
    total_value = 0
    current_storage = 0

    # BUY (Injection)
    for date in injection_dates:
        price = estimate_price(date)

        volume = min(injection_rate, volume_per_trade)
        if current_storage + volume > max_storage_volume:
            volume = max_storage_volume - current_storage

        total_value -= price * volume
        total_value -= injection_cost

        current_storage += volume

    # SELL (Withdrawal)
    for date in withdrawal_dates:
        price = estimate_price(date)

        volume = min(withdrawal_rate, current_storage)

        total_value += price * volume
        total_value -= withdrawal_cost

        current_storage -= volume

    # STORAGE COST
    if injection_dates and withdrawal_dates:
        start = pd.to_datetime(min(injection_dates))
        end = pd.to_datetime(max(withdrawal_dates))

        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_value -= months * storage_cost_per_month

    return total_value

# ==============================
# 5. TEST THE MODEL
# ==============================

value = price_storage_contract(
    injection_dates=["2024-05-31", "2024-06-30"],   # summer (cheap)
    withdrawal_dates=["2024-12-31", "2025-01-31"], # winter (expensive)
    volume_per_trade=1_000_000,
    max_storage_volume=2_000_000,
    injection_rate=1_000_000,
    withdrawal_rate=1_000_000,
    storage_cost_per_month=100_000,
    injection_cost=10_000,
    withdrawal_cost=10_000
)

print(f"Contract Value: {value:,.2f}")