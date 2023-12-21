# import pandas as pd
# from plyer import notification
# import time 

# csv_file_path=r"D:\pollutiondata_Final (1).csv"

# dangerous_levels={
# 'CO':200,
# 'NH3':80,
# 'NO2':300,
# 'PM10':450,
# 'PM2.5':450,
# 'OZONE':270,
# 'SO2':100,
# 'AQI':400,
# }
# def check_csv_for_dangerous_values():
#     try:
#         df = pd.read_csv(csv_file_path)

#         # Iterate through rows and columns to check for dangerous values
#         for index, row in df.iterrows():
#             for pollutant, threshold in dangerous_levels.items():
#                 value = row[pollutant]
#                 if value > threshold:
#                     timestamp = row['timestamp']
                   
                    
#                     notification_title = f"Dangerous {pollutant} Level Alert"
#                     notification_text = f"High {pollutant} value of {value} detected at {timestamp}."
#                     aqi_text = f"{pollutant}"
                    
#                     notification_message = f"{notification_text}\n"
                    
#                     notification.notify(
#                         title=notification_title,
#                         message=notification_message,
#                         app_name="CSV Monitor",
#                     )
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# # Set the monitoring interval (in seconds)
# monitoring_interval = 60  # Check every 60 seconds

# while True:
#     check_csv_for_dangerous_values()
#     time.sleep(monitoring_interval)




# import pandas as pd
# from plyer import notification
# from datetime import datetime

# csv_file_path=r"D:\pollutiondata_Final (1).csv"

# dangerous_level={
#     'CO':200,
#     'NH3':80,
#     'NO2':300,
#     'PM10':450,
#     'PM2.5':450,
#     'OZONE':270,
#     'SO2':100,
#     'AQI':400,
# }

# def check_csv_for_dangerous_values():
#     try:
#         df = pd.read_csv('pollutiondata_Final4.csv',low_memory=False)

#         # Iterate through rows and columns to check for dangerous values
#         for index, row in df.iterrows():
#             for pollutant, threshold in dangerous_level.items():
#                 value = row[pollutant]
#                 if float(value) > threshold:
#                     timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + row['Station']
#                     print(timestamp)
#                     # aqi = calculate_aqi(value, pollutant)
                    
#                     notification_title = f"Dangerous {pollutant} Level Alert"
#                     notification_text = f"High {pollutant} value of {value} detected at {timestamp}."
#                     aqi_text = f"{pollutant}"
                    
#                     notification_message = f"{notification_text}\n{aqi_text}"
                    
#                     notification.notify(
#                         title=notification_title,
#                         message=notification_message,
#                         app_name="CSV Monitor",
#                     )
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# # Set the monitoring interval (in seconds)
# monitoring_interval = 60  # Check every 60 seconds

# while True:
#     check_csv_for_dangerous_values()
#     time.sleep(monitoring_interval)





import pandas as pd
from plyer import notification
from datetime import datetime
import time   

csv_file_path = r"D:\pollutiondata_Final (1).csv"

dangerous_level = {
    'CO': 200,
    'NH3': 80,
    'NO2': 300,
    'PM10': 450,
    'PM2.5': 450,
    'OZONE': 270,
    'SO2': 100,
    'AQI': 400,
}


def calculate_aqi(value, pollutant):
    pass

def check_csv_for_dangerous_values():
    try: 
        df = pd.read_csv(r"D:\pollutiondata_Final (1).csv", low_memory=False)

        for index, row in df.iterrows():
            for pollutant, threshold in dangerous_level.items():
                value = row[pollutant]
                if float(value) > threshold:
                    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + row['Station']
                    print(timestamp)
                    aqi = calculate_aqi(value, pollutant)  
                    
                    notification_title = f"Dangerous {pollutant} Level Alert"
                    notification_text = f"High {pollutant} value of {value} detected at {timestamp}."
                    aqi_text = f"AQI: {aqi}" if aqi is not None else ""
                    
                    notification_message = f"{notification_text}\n{aqi_text}"
                    
                    notification.notify(
                        title=notification_title,
                        message=notification_message,
                        app_name="CSV Monitor",
                    )
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Set the monitoring interval (in seconds)
monitoring_interval = 60  # Check every 60 seconds

while True:
    check_csv_for_dangerous_values()
    time.sleep(monitoring_interval)





