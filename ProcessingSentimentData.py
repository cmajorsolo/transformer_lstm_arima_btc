from os import listdir
from os.path import join 
import pandas as pd 

def combine_all_sentiment_data(file_path, file_name_prefix, key_column, output_file_name):
    # read files in file_path with file_name_prefix
    file_names = [f for f in listdir(file_path) if f.startswith(file_name_prefix)]
    # combine all the dfs into one df
    sentiment_df = pd.DataFrame()
    for file_name in file_names:
        file_name = join(file_path, file_name)
        df = pd.read_csv(file_name)
        sentiment_df = sentiment_df.append(df)
    
    print('lenght of df is {}'.format(len(sentiment_df)))
    # order by datetime 
    sentiment_df = sentiment_df.sort_values(by=[key_column])
    # set df key
    sentiment_df.set_index(key_column)
    # write the df to a csv file
    sentiment_df.to_csv(output_file_name, index=False)

# sentiment_df = combine_all_sentiment_data('./Data/Sentiment_data', 'eth_sentiment_', 'Datetime', 'eth_sentiment.csv')

def merge_csv_based_on_key(sentiment_csv_path, price_data_path, to_csv_file_name):
    # looking for duplciated rows in sentiment csv file 
    # sentiment_df = pd.read_csv('./Data/Sentiment_data/btc_sentiment.csv')
    # sentiment_df = pd.read_csv(sentiment_csv_path)
    # dates = sentiment_df['Datetime']
    
    # Got duplicated rows and remove the rows from the csv.
    # duplicate = dates[dates.duplicated(keep='last')]
    # print('duplicates are ')
    # print(duplicate)
    

    # merge files on date time column
    # price_data = pd.read_csv('./Data/BTC_3hrs_History_20150101_20210427.csv')
    price_data = pd.read_csv(price_data_path)
    price_data['time_period_start'] = pd.to_datetime(price_data['time_period_start'], format='%Y-%m-%d %H:%M').dt.date
    print(price_data.head(5))
    # sentiment_data = pd.read_csv('./Data/Sentiment_data/btc_sentiment.csv')
    sentiment_data = pd.read_csv(sentiment_csv_path)
    sentiment_data['Datetime'] = pd.to_datetime(sentiment_data['Datetime'], format='%Y-%m-%d %H:%M').dt.date
    print(sentiment_data.head(5))

    final = price_data.merge(sentiment_data, how='left', left_on='time_period_start', right_on='Datetime')
    # final.to_csv('./Data/BTC_merged_with_sentiment_data.csv', index=False)
    final.to_csv(to_csv_file_name, index=False)
    print(final.head(5))

# merge_csv_based_on_key('./Data/Sentiment_data/eth_sentiment.csv', './Data/ETH_3rs_History_20150101_20210427.csv', './Data/ETH_merged_with_sentiment_data.csv')
