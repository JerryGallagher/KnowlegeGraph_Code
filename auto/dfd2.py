import pandas as pd

# Sample data as a dictionary
data = {
    'Topic': ['DAO and SQL Mapping for Account and Product Management', 'User Account Details and Cart Management', 'Account Update Operations',  'Order Details and Validation for Billing and Shipping' , 'Address Management for Billing and Shipping', 'Payment Information and Order Completion' ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)
