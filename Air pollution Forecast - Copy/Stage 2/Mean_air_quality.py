import pandas as pd

# Load your dataset
file_path = r"F:\Education\COLLEGE\PROGRAMING\Python\Codes\Air pollution Forecast - Copy\Air pollution Forecast - Copy\Knowledge Park - V, Greater Noida - UPPCB_pollutant_data.csv"
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format with errors='coerce'
# df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')

# Check for rows with NaT (indicating parsing errors)
rows_with_errors = df[df['Date'].isna()]
print("Rows with date parsing errors:")
print(rows_with_errors)
columns_to_mean = ['PM10', 'PM2.5', 'SO2', 'OZONE', 'CO', 'NO2', 'NH3', 'AQI']
# Continue with your data processing or analysis
df_mean = df.groupby('Date')[columns_to_mean].mean().reset_index()
# Save the result to a new CSV file
output_file_path = 'path_to_output_csv_file.csv'
df_mean.to_csv(output_file_path, index=False)

# Display the first few rows of the new DataFrame
print(df_mean.head())
