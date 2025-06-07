import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths
base_dir = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(base_dir, '../data/train.csv'), parse_dates=["Date"])
test = pd.read_csv(os.path.join(base_dir, '../data/test.csv'), parse_dates=["Date"])
store = pd.read_csv(os.path.join(base_dir, '../data/store.csv'))

# Merge with store info
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
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["PromoInterval"] = df["PromoInterval"].fillna("")
    df["IsPromoMonth"] = df.apply(
        lambda row: 1 if row["Month"] in [1, 4, 7, 10] and row["PromoInterval"] != "" else 0, axis=1
    )
    df["CompetitionOpenSince"] = 12 * (df["Year"] - df["CompetitionOpenSinceYear"]) + \
                                  (df["Month"] - df["CompetitionOpenSinceMonth"])
    df["CompetitionOpenSince"] = df["CompetitionOpenSince"].apply(lambda x: x if x > 0 else 0)
    return df

train = add_features(train)
test = add_features(test)

# Encode categorical variables
categorical_cols = ["StoreType", "Assortment", "StateHoliday"]
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Filter training data
train = train[(train["Open"] == 1) & (train["Sales"] > 0)]

# Features and target
features = ["Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "WeekOfYear",
            "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance",
            "CompetitionOpenSince", "IsPromoMonth"]

X_train = train[features]
y_train = train["Sales"]
X_test = test[features]

# Train LightGBM model
logging.info("Training model...")
model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train)

# Cross-validation score
cv_score = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3)
logging.info("Cross-validated RMSE: %.2f", -cv_score.mean())

# Built-in Feature Importance Plot
importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../data/feature_importance_plot.png"))
plt.close()

# Predict and save results
preds = model.predict(X_test)
preds[preds < 0] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": preds})
submission.to_csv(os.path.join(base_dir, '../data/submission.csv'), index=False)

logging.info("Submission file saved to ../data/submission.csv")
logging.info("Feature importance plot saved to ../data/feature_importance_plot.png")
