import pandas as pd
import json

# Example JSON data
json_data = '{"employee": [{"id": "101", "name": "John", "department": "Sales"}, {"id": "102", "name": "Alice", "department": "Marketing"}]}'

# Load JSON data into a DataFrame
data = json.loads(json_data)
df = pd.DataFrame(data['employee'])

# Print the DataFrame
print(df)