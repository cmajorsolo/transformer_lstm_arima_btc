import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def preprocess_train_test_val(data_path, normalization=True):
    # btc_path = './Data/BTC_Daily_History_20150101_20210427.csv'
    # btc_path = './Data/BTC_3hrs_History_20150101_20210427.csv'
    df = pd.read_csv(data_path, delimiter=',', usecols=['time_period_start', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded'])
    df = df.rename(columns={"time_period_start": "Date", "price_open": "Open", "price_high": "High", "price_low": "Low", "price_close": "Close", "volume_traded": "Volume"})
    # Replace 0 to avoid dividing by 0 later on
    df['Volume'].replace(to_replace=0, method='ffill', inplace=True) 
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')#.dt.date
    df.dropna(how='any', axis=0, inplace=True)

    '''Create indexes to split dataset'''
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series
    '''Create training, validation and test split'''

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    # Remove date column
    df_train.drop(columns=['Date'], inplace=True)
    df_val.drop(columns=['Date'], inplace=True)
    df_test.drop(columns=['Date'], inplace=True)

    if normalization:
        # Data normalization 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_train[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df_train[['Open', 'High', 'Low', 'Close']])
        df_train[['Volume']] = scaler.fit_transform(df_train[['Volume']])

        df_val[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df_val[['Open', 'High', 'Low', 'Close']])
        df_val[['Volume']] = scaler.fit_transform(df_val[['Volume']])

        df_test[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df_test[['Open', 'High', 'Low', 'Close']])
        df_test[['Volume']] = scaler.fit_transform(df_test[['Volume']])

    # Convert pandas columns into arrays
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    print('Training data shape: {}'.format(train_data.shape))
    print('Validation data shape: {}'.format(val_data.shape))
    print('Test data shape: {}'.format(test_data.shape))
    return df_train, train_data, df_val, val_data, df_test, test_data

def preprocess_train_val_test_Xy(train_data, val_data, test_data, seq_len=240):
    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 240 df-rows
        y_train.append(train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row 240+1
    X_train, y_train = np.array(X_train), np.array(y_train)
    print('Training set shape', X_train.shape, y_train.shape)

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i-seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)
    print('Validation set shape', X_val.shape, y_val.shape)

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i-seq_len:i])
        y_test.append(test_data[:, 3][i])    
    X_test, y_test = np.array(X_test), np.array(y_test)
    print('Testing set shape' ,X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_df(df, currency='BTC'):
    fig = plt.figure(figsize=(15,10))
    st = fig.suptitle("{} Close Price and Volume".format(currency), fontsize=20)
    st.set_y(0.92)

    ax1 = fig.add_subplot(211)
    ax1.plot(df['Close'], label='{} Close Price'.format(currency))
    ax1.set_xticks(range(0, df.shape[0], 1464))
    ax1.set_xticklabels(df['Date'].loc[::1464])
    ax1.set_ylabel('Close Price', fontsize=18)
    ax1.legend(loc="upper left", fontsize=12)

    ax2 = fig.add_subplot(212)
    ax2.plot(df['Volume'], label='{} Volume'.format(currency))
    ax2.set_xticks(range(0, df.shape[0], 1464))
    ax2.set_xticklabels(df['Close'].loc[::1464])
    ax2.set_ylabel('Volume', fontsize=18)
    ax2.legend(loc="upper left", fontsize=12)

def plot_train_val_test(train_data, df_train, val_data, df_val, test_data, df_test):
    fig = plt.figure(figsize=(15,12))
    st = fig.suptitle("Data Separation", fontsize=20)
    st.set_y(0.6)

    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(train_data.shape[0]), df_train['Close'], label='Training data')

    ax1.plot(np.arange(train_data.shape[0], 
                    train_data.shape[0]+val_data.shape[0]), df_val['Close'], label='Validation data')

    ax1.plot(np.arange(train_data.shape[0]+val_data.shape[0], 
                    train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['Close'], label='Test data')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Closing Returns')
    ax1.set_title("Close Price", fontsize=18)
    ax1.legend(loc="best", fontsize=12)

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(train_data.shape[0]), df_train['Volume'], label='Training data')

    ax2.plot(np.arange(train_data.shape[0], 
                    train_data.shape[0]+val_data.shape[0]), df_val['Volume'], label='Validation data')

    ax2.plot(np.arange(train_data.shape[0]+val_data.shape[0], 
                    train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['Volume'], label='Test data')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Volume Changes')
    ax2.set_title("Volume", fontsize=18)
    ax2.legend(loc="best", fontsize=12)

def plot_result(train_data, train_pred, val_data, val_pred, test_data, test_pred, seq_len=240, fig_name='Pred_vs_Reality.png'):
    '''Display results'''
    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
    st.set_y(0.92)

    #Plot training data results
    ax11 = fig.add_subplot(311)
    ax11.plot(train_data[:, 3], label='Closing Returns')
    ax11.plot(np.arange(seq_len, train_pred.shape[0]+seq_len), train_pred, linewidth=3, label='Predicted Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    #Plot validation data results
    ax21 = fig.add_subplot(312)
    ax21.plot(val_data[:, 3], label='Closing Returns')
    ax21.plot(np.arange(seq_len, val_pred.shape[0]+seq_len), val_pred, linewidth=3, label='Predicted Closing Returns')
    ax21.set_title("Validation Data", fontsize=18)
    ax21.set_xlabel('Date')
    ax21.set_ylabel('Closing Returns')
    ax21.legend(loc="best", fontsize=12)

    #Plot test data results
    ax31 = fig.add_subplot(313)
    ax31.plot(test_data[:, 3], label='Closing Returns')
    ax31.plot(np.arange(seq_len, test_pred.shape[0]+seq_len), test_pred, linewidth=3, label='Predicted Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('Closing Returns')
    ax31.legend(loc="best", fontsize=12)

    plt.savefig(fig_name)

def plot_history(history, fig_name='history_loss.png'):
    '''Display model metrics'''
    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Transformer + TimeEmbedding Model Metrics", fontsize=22)
    st.set_y(0.92)

    #Plot model loss
    ax1 = fig.add_subplot(311)
    ax1.plot(history.history['loss'], label='Training loss (MSE)')
    ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12)

    #Plot MAE
    ax2 = fig.add_subplot(312)
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean average error (MAE)')
    ax2.legend(loc="best", fontsize=12)

    #Plot MAPE
    ax3 = fig.add_subplot(313)
    ax3.plot(history.history['mape'], label='Training MAPE')
    ax3.plot(history.history['val_mape'], label='Validation MAPE')
    ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean average percentage error (MAPE)')
    ax3.legend(loc="best", fontsize=12)

    plt.savefig(fig_name)

