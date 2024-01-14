import pandas as pd
import ast

# Function to duplicate rows based on the number of '1' occurrences in Tags
def duplicate_rows(row):
    tokens = ast.literal_eval(row['English Review'])
    
    # Check for NaN in 'Tags' column
    tags = ast.literal_eval(row['Tags']) if pd.notna(row['Tags']) else []
    
    return pd.DataFrame({'English Review': [tokens] * tags.count(1), 'Tags': [tags] * tags.count(1)})

# Input and output file paths
input_file_path = 'NasiLemakReviewss.csv'
output_file_path = 'NasiLemakReviewTags.csv'

# Read the CSV file
df = pd.read_csv(input_file_path)

# Apply the duplicate_rows function to each row and store the resulting DataFrames in a list
duplicated_data = [duplicate_rows(row) for _, row in df.iterrows()]

# Concatenate the list of DataFrames
result_df = pd.concat(duplicated_data, ignore_index=True)

# Save the result to a new CSV file
result_df.to_csv(output_file_path, index=False)
print(f'Duplicated rows saved to {output_file_path}')
