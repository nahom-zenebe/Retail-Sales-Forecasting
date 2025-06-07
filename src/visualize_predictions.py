import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Load data
base_dir = os.path.dirname(__file__)
submission = pd.read_csv(os.path.join(base_dir, "../data/submission.csv"))
test = pd.read_csv(os.path.join(base_dir, "../data/test.csv"), parse_dates=["Date"])
df = pd.merge(test, submission, on="Id")

# Plot 1: Daily sales
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()
plt.figure(figsize=(14, 6))
sns.lineplot(x="Date", y="Sales", data=daily_sales, color="blue", linewidth=2)
plt.title("Predicted Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../data/daily_sales_plot.png"))
plt.show()

# Plot 2: Top 20 stores
top_stores = df.groupby("Store")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(20)
plt.figure(figsize=(12,6))
sns.barplot(x="Store", y="Sales", data=top_stores)
plt.title("Top 20 Stores by Predicted Sales")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../data/top_stores_plot.png"))
plt.show()

# Plot 3: Sales by Day of Week
df["DayOfWeek"] = df["Date"].dt.dayofweek
dow = df.groupby("DayOfWeek")["Sales"].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x="DayOfWeek", y="Sales", data=dow)
plt.title("Average Sales by Day of Week")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../data/dayofweek_plot.png"))
plt.show()
