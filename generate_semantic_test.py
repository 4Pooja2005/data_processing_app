import pandas as pd

# Creating a dataset where most rows are valid semantic types but some are corrupt
data = {
    "ID": [1, 2, 3, 4, 5],
    "User_Contact": [
        "john@example.com",
        "sarah@test.org",
        "NOT_AN_EMAIL_LOL", # Invalid
        "mike@company.net",
        "12345" # Invalid
    ],
    "SignUp_Info": [
        "2023-01-15",
        "2023-02-28",
        "2023-03-10",
        "March 15th", # Invalid format based on regex
        "2023-04-05"
    ],
    "Revenue": [
        "$100.50",
        "$200.00",
        "$300.25",
        "Zero Dollars", # Invalid
        "$500.00"
    ]
}

df = pd.DataFrame(data)
df.to_csv("semantic_test_data.csv", index=False)
print("Generated semantic_test_data.csv")
