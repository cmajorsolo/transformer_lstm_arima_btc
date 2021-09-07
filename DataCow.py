'''
Downloading data and preprocessing data
'''
import json
import requests
import pandas as pd
import datetime as dt

file_name = "ETH_3rs_History_20150101_20210427.csv"

api_key = "C665DC77-1EC6-4205-B4BF-808C86A6B273"
url = "https://rest.coinapi.io/v1/ohlcv/ETH/USD/history?period_id=3HRS&time_start=2015-01-01T00:00:00&time_end=2021-04-27T23:59:00&limit=100000"
# url = "https://rest.coinapi.io/v1/ohlcv/BTC/GBP/history?period_id=1DAY&time_start=2015-01-01T00:00:00&time_end=2020-10-31T23:59:00"
headers = {"X-CoinAPI-Key" : api_key}
response = requests.get(url, headers = headers)

if(response.status_code == 429):
    # API response
    print("Too many requests.")

coin_data  = json.loads(response.text)
with open('eth_history.json', 'wb') as f:
    f.write(response.content)
btc_data = pd.DataFrame(coin_data)
# btc_data["Start Time"] = pd.to_datetime(btc_data["Start Time"])
# btc_data["Day of the Week"] = btc_data['Start Time'].dt.dayofweek

# def number_to_day(number):
#     days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#     return days[number]

# btc_data["Day of the Week"] = btc_data["Day of the Week"].apply(number_to_day)
btc_data.to_csv(file_name, index = False)
print(btc_data.head())