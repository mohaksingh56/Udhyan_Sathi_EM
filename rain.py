import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r"D:\output.csv")

features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM'
]

target = 'RainTomorrow'

data = data.dropna(subset=[target])

data = data[features + [target]]
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

xgb_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    XGBClassifier()
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) * 100 
print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {accuracy:.2f}%')  

