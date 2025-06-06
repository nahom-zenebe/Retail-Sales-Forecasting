import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import os

print("Current working directory:", os.getcwd())

# Load datasets
train = pd.read_csv("C:/Users/Abel Tesfa/Desktop/retail_sales/Retail-Sales-Forecasting/train.csv", parse_dates=["Date"])
test = pd.read_csv("C:/Users/Abel Tesfa/Desktop/retail_sales/Retail-Sales-Forecasting/test.csv", parse_dates=["Date"])
store = pd.read_csv("C:/Users/Abel Tesfa/Desktop/retail_sales/Retail-Sales-Forecasting/store.csv")

# Merge store info
train = train.merge(store, on="Store", how="left")
test = test.merge(store, on="Store", how="left")

# Fill missing values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Feature engineering
def add_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["PromoInterval"] = df["PromoInterval"].fillna("")
    df["IsPromoMonth"] = df.apply(lambda row: 1 if row["Month"] in 
                                  [int(m) for m in [1,4,7,10] if row["PromoInterval"] != ""] else 0, axis=1)
    return df

train = add_features(train)
test = add_features(test)

# Encode categoricals
categorical_cols = ["StoreType", "Assortment", "StateHoliday"]
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Only train on open days with sales > 0
train = train[(train["Open"] == 1) & (train["Sales"] > 0)]

# Features and target
features = ["Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "WeekOfYear",
            "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "IsPromoMonth"]

X_train = train[features]
y_train = train["Sales"]
X_test = test[features]

# Train LightGBM
model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
preds[preds < 0] = 0  # No negative sales

# Prepare submission
submission = pd.DataFrame({"Id": test["Id"], "Sales": preds})
submission.to_csv("submission.csv", index=False)

print("Loaded data")
print("Merged store info")
print("Filled missing values")
print("Added features")
print("Encoded categoricals")
print("Filtered training data")
print("Prepared features and target")
print("Trained model")
print("Made predictions")
print("Saving submission file")

print("X_test shape:", X_test.shape)
print("test['Id'] shape:", test['Id'].shape)
print("preds shape:", preds.shape)
