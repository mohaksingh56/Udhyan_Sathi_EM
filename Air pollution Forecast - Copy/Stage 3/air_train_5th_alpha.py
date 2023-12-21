import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector

# Load your dataset
dataset_path = r"F:\Education\COLLEGE\PROGRAMING\Python\PROJECTS\PollutionDataAnalysisProject\Platinum\pollutiondata_Final.csv"
dataset = pd.read_csv(dataset_path)

db_config = {
            'host': '127.0.0.1',
            'user': 'root',
            'password': 'Impetus@123',
            'database': 'UdyaanSaathiData'
        }
conn = mysql.connector.connect(**db_config)

# Create a MySQL cursor
cursor = conn.cursor()

# Selected pollutants
pollutants = ['PM10', 'PM2.5', 'NO2', 'NH3', 'OZONE', 'CO', 'SO2', 'AQI', 'Date']

# Iterate over each unique station in the dataset
for station in dataset['Station'].unique():
    print(f"Processing data for station: {station}")
    
    # Filter data for the specified station and pollutants
    filtered_data = dataset[dataset['Station'] == station][['Station'] + pollutants]

    if not filtered_data.empty:
        # Save the filtered data to a new CSV file
        # filtered_data.to_csv(f'{station}_pollutant_data.csv', index=False)
        # print(f"Filtered data for {station} saved to {station}_pollutant_data.csv.")
        
        # Continue with data processing and machine learning
        # ...

        # For example, applying machine learning on the filtered data
        features = ['PM10', 'PM2.5', 'SO2', 'OZONE', 'CO', 'NO2', 'NH3']
        target = 'AQI'

        X = filtered_data[features]
        y = filtered_data[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.insert(X_scaled, 0, 1, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error for {station}: {mse}')
        print(f'R-squared Score for {station}: {r2}')

        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual AQI Values')
        plt.ylabel('Predicted AQI Values')
        plt.title(f'Actual vs. Predicted AQI Values for {station}')
        # plt.show()

        latest_data = X_scaled[-1].reshape(1, -1)
        next_days_predictions = []
        for _ in range(5):
            prediction = rf_model.predict(latest_data)[0]
            next_days_predictions.append(prediction)
            latest_data = np.roll(latest_data, -1)
            latest_data[0, -1] = prediction

        days_columns = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']
        predictions_dict = {'Station': station}
        predictions_dict.update(zip(days_columns, next_days_predictions))

        # Convert predictions_dict to a DataFrame
        predictions_df = pd.DataFrame(predictions_dict, index=[0])

        # Convert NaN values to NULL for MySQL compatibility
        predictions_df = predictions_df.where(pd.notna(predictions_df), None)

        # Convert DataFrame to a list of tuples
        values = [tuple(row) for row in predictions_df.values]

        # Prepare the SQL query
        insert_query = f"INSERT INTO MLData ({', '.join(predictions_df.columns)}) VALUES ({', '.join(['%s'] * len(predictions_df.columns))})"

        # Execute the SQL query with the data
        cursor.executemany(insert_query, values)

        # Commit the changes to the database
        conn.commit()
        print(f'Predictions for the next 5 days for {station}:', next_days_predictions)

    else:
        print(f"No data available for the specified station: {station}")
cursor.close()
conn.close()