import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
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


user_input = {}
for feature in features:
    if feature in data.select_dtypes(include=['object']).columns:
 
        user_input[feature] = label_encoder.transform(
            [input(f"Enter value for {feature}: ")])[0] if label_encoder.classes_.__contains__(input(f"Enter value for {feature}: ")) else len(label_encoder.classes_)
    else:
        user_input[feature] = float(input(f"Enter value for {feature}: "))


user_df = pd.DataFrame([user_input])

user_pred = xgb_model.predict(user_df)


print(f"The model predicts(0-> No, 1-> Yes): {user_pred[0]} ")
