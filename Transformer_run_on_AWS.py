import argparse, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))

import s3fs
fs = s3fs.S3FileSystem()

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

from Time2Vector import Time2Vector
from Transformer import TransformerEncoder, SingleAttention, MultiAttention
from tensorflow.keras.utils import multi_gpu_model
from keras import backend as K

def read_data_from_s3(uri):
    data = pd.DataFrame()
    with fs.open(uri) as f:
        data = pd.read_csv(f)
    return data

def create_model():
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    
    '''Construct model'''
    in_seq = tf.keras.Input(shape=(seq_len, 5))
    x = time_embedding(in_seq)
    x = tf.keras.layers.Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=in_seq, outputs=out)
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=240)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--learning-rate', type=float, default=0.01)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    seq_len    = args.seq_len

    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation

    d_k = 256
    d_v = 256
    n_heads = 12
    ff_dim = 256

    model_name = 'transformer'

    # Getting data from S3
    train_data_uri = 's3://cmajorsolo-transformerbucket/data/btc_train.csv'
    test_data_uri = 's3://cmajorsolo-transformerbucket/data/btc_test.csv'
    val_data_uri = 's3://cmajorsolo-transformerbucket/data/btc_val.csv'

    train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_data = read_data_from_s3(train_data_uri)
    val_data = read_data_from_s3(val_data)
    test_data = read_data_from_s3(test_data)  


    # Training data
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 240 df-rows
        y_train.append(train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row 240+1
    X_train, y_train = np.array(X_train), np.array(y_train)

    ###############################################################################

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i-seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)

    ###############################################################################

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i-seq_len:i])
        y_test.append(test_data[:, 3][i])    
    X_test, y_test = np.array(X_test), np.array(y_test)

    model = create_model()
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
  
    
    callback = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        callbacks=[callback],
                        validation_data=(X_val, y_val))  
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})

