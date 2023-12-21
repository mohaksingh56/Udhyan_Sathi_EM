import pandas as pd

# Load your dataset
dataset = pd.read_csv(r"F:\Education\COLLEGE\PROGRAMING\Python\PROJECTS\PollutionDataAnalysisProject\Platinum\pollutiondata_Final.csv") # Replace 'your_dataset.csv' with the actual file path

# User input for the city
station = input("Enter the Station: ")

# Selected pollutants
pollutants = ['PM10', 'PM2.5', 'NO2', 'NH3', 'OZONE', 'CO', 'SO2', 'AQI','Date']

# Filter data for the specified city and pollutants
filtered_data = dataset[dataset['Station'] == station][['Station'] + pollutants]

if filtered_data.empty:
    print(f"No data available for the specified city: {station}")
else:
    # Save the filtered data to a new CSV file
    filtered_data.to_csv(f'{station}_pollutant_data.csv', index=False)
    print(f"Filtered data for {station} saved to {station}_pollutant_data.csv.")
