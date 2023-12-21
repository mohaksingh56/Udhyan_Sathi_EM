import os
import pandas as pd

# Load your dataset
dataset = pd.read_csv(r"F:\Education\COLLEGE\PROGRAMING\Python\PROJECTS\PollutionDataAnalysisProject\Platinum\pollutiondata_Final.csv")

# Selected pollutants
pollutants = ['PM10', 'PM2.5', 'NO2', 'NH3', 'OZONE', 'CO', 'SO2', 'AQI', 'Date']

# Base folder path
base_folder = r"F:\Education\COLLEGE\PROGRAMING\Python\PROJECTS\PollutionDataAnalysisProject\ML"

# Get unique stations in the dataset
all_stations = dataset['Station'].unique()

# Iterate through all stations
for station in all_stations:
    # Create a folder for each station if it doesn't exist
    station_folder = os.path.join(base_folder, station)
    os.makedirs(station_folder, exist_ok=True)

    # Filter data for the specified station and pollutants
    filtered_data = dataset[dataset['Station'] == station][['Station'] + pollutants]

    if not filtered_data.empty:
        # Save the filtered data to a new CSV file in the station folder
        csv_path = os.path.join(station_folder, f'{station}_pollutant_data.csv')
        filtered_data.to_csv(csv_path, index=False)
        print(f"Filtered data for {station} saved to {csv_path}.")
    else:
        print(f"No data available for the specified city: {station}")
