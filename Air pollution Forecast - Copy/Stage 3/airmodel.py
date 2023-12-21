from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Ridge
from sklearn.linear_model import  Lasso
from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

air_df = pd.read_csv(r"F:\Education\COLLEGE\PROGRAMING\Python\Codes\Air pollution Forecast - Copy\Air pollution Forecast - Copy\path_to_output_csv_file.csv")

# Try parsing the date with infer_datetime_format
air_df['Date'] = pd.to_datetime(air_df['Date'], infer_datetime_format=True, errors='coerce')

# Check for errors after parsing
errors = air_df['Date'][air_df['Date'].isnull()]
# if not errors.empty:
    # print("Rows with date parsing errors:")
    # print(errors)

# Rest of your code
air_df['AQI_lag'] = air_df['AQI'].shift(1)
# Add other columns or processing as needed

# Print the resulting DataFrame to check the changes
# print(air_df.head())


features = ['PM10', 'PM2.5', 'SO2', 'OZONE', 'CO', 'NO2', 'NH3']
target = 'AQI'


X = air_df[features]
y = air_df[target]

X = air_df[features]
y = air_df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = np.insert(X_scaled, 0, 1, axis=1)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R-squared Score: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual AQI Values')
plt.ylabel('Predicted AQI Values')
plt.title('Actual vs. Predicted AQI Values')
# plt.show()


latest_data = X_scaled[-1].reshape(1, -1)
next_days_predictions = []
for _ in range(5):
    prediction = rf_model.predict(latest_data)[0]
    next_days_predictions.append(prediction)
    # Update the latest_data for the next prediction
    latest_data = np.roll(latest_data, -1)
    latest_data[0, -1] = prediction

print('Predictions for the next 5 days:', next_days_predictions)