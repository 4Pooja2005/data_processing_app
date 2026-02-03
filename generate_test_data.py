import pandas as pd
import numpy as np
import os
import random

# Create directory
os.makedirs("test_data", exist_ok=True)

# 1. Clean Users: Baseline valid data
data1 = {
    "ID": range(1, 41),
    "Name": [f"User_{i}" for i in range(1, 41)],
    "Role": [random.choice(["Admin", "Editor", "Viewer"]) for _ in range(40)],
    "Age": [random.randint(20, 60) for _ in range(40)],
    "Active": [random.choice([True, False]) for _ in range(40)]
}
pd.DataFrame(data1).to_csv("test_data/1_clean_users.csv", index=False)

# 2. Missing Values Sales (Excel): Tests handling of NaNs
data2 = {
    "Product": [f"Prod_{i}" for i in range(1, 41)],
    "Quantity": [random.choice([1, 5, 10, np.nan]) for _ in range(40)],
    "Price": [random.choice([10.5, 99.99, np.nan, 5.0]) for _ in range(40)],
    "Region": ["North", "South", "East", "West"] * 10
}
pd.DataFrame(data2).to_excel("test_data/2_missing_values_sales.xlsx", index=False)

# 3. Duplicates Logs: Tests remove_duplicates
data3 = {
    "Timestamp": pd.date_range(start="2023-01-01", periods=10, freq="h").tolist() * 4,
    "Event": ["Login", "Logout", "View", "Click", "Error"] * 2 * 4,
    "UserID": [101, 102, 103, 104, 105] * 2 * 4
}
pd.DataFrame(data3).to_csv("test_data/3_duplicates_logs.csv", index=False)

# 4. Messy Text Inventory: Tests standardize_data (strip, lowercase)
item_names = ["  apple ", "Apple", "APPLE  ", " baNana", "Orange", "orange ", "  GRAPE  ", "Melon"] * 5
data4 = {
    "ItemName": item_names,
    "Category": ["Fruit"] * 40,
    "Stock": [random.randint(0, 100) for _ in range(40)]
}
pd.DataFrame(data4).to_csv("test_data/4_messy_text_inventory.csv", index=False)

# 5. Mixed Types Feedback: Tests to_numeric errors/coercion
ratings = [1, 5, "Five", 3.5, "Three", 4, 2, "N/A", 5, 1] * 4
data5 = {
    "Rating": ratings,
    "Comment": [f"Comment {i}" for i in range(40)]
}
pd.DataFrame(data5).to_csv("test_data/5_mixed_types_feedback.csv", index=False)

# 6. Large Numbers Finance: Tests numeric precision/display
data6 = {
    "TransactionID": range(1000, 1040),
    "Amount": [random.uniform(1e6, 1e9) for _ in range(40)],
    "Account": [random.randint(100000000000, 999999999999) for _ in range(40)]
}
pd.DataFrame(data6).to_csv("test_data/6_large_numbers_finance.csv", index=False)

# 7. Dates Events: Tests date parsing (which is missing)
dates = ["2023-01-01", "01/02/2023", "Mar 3, 2023", "2023.04.05", "May-06-2023"] * 8
data7 = {
    "EventName": [f"Event_{i}" for i in range(40)],
    "Date": dates
}
pd.DataFrame(data7).to_csv("test_data/7_dates_events.csv", index=False)

# 8. Merge Part 1 Employees: For merge test
data8 = {
    "ID": range(1, 21),
    "Name": [f"Emp_{i}" for i in range(1, 21)],
    "Dept": ["HR", "IT", "Sales", "Marketing"] * 5
}
pd.DataFrame(data8).to_csv("test_data/8_merge_part1_employees.csv", index=False)

# 9. Merge Part 2 Employees: For merge test (overlaps)
data9 = {
    "ID": range(15, 35), # Overlap 15-20
    "Name": [f"Emp_{i}" for i in range(15, 35)],
    "Dept": ["HR", "IT", "Sales", "Marketing"] * 5
}
pd.DataFrame(data9).to_csv("test_data/9_merge_part2_employees.csv", index=False)

# 10. Security Edge Cases: CSV Injection, large strings
malicious = ["=1+1", "@SUM(1+1)", "-1+1", "+1+1", "Safe", "Normal", "Drop Tables", "<script>alert(1)</script>"] * 5
long_str = ["A" * 1000, "B" * 5000, "Short", "Normal"] * 10
data10 = {
    "Input": malicious,
    "Notes": long_str
}
pd.DataFrame(data10).to_csv("test_data/10_security_edge_cases.csv", index=False)

print("Test datasets generated in 'test_data/' directory.")
