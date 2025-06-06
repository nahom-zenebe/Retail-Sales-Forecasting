import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional: set a style
sns.set(style="whitegrid")

# Load prediction submission file
submission_path = "submission.csv"
submission = pd.read_csv(submission_path)

# Load test data to retrieve dates
test_path = "C:/Users/Abel Tesfa/Desktop/retail_sales/Retail-Sales-Forecasting/test.csv"
test = pd.read_csv(test_path, parse_dates=["Date"])

# Merge predicted sales with test data
df = pd.merge(test, submission, on="Id")

# Group by Date to see total predicted sales over time
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

# Plot total sales over time
plt.figure(figsize=(14, 6))
sns.lineplot(x="Date", y="Sales", data=daily_sales, color="blue", linewidth=2)
plt.title("Predicted Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optionally, save the plot
# plt.savefig("predicted_sales_plot.png")
