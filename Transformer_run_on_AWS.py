import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow.python.keras.backend import _get_available_gpus 
import argparse, os
import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
import json

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from Time2Vector import Time2Vector
from Transformer import TransformerEncoder, SingleAttention, MultiAttention
from tensorflow.keras.utils import multi_gpu_model

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
seq_len = 240
# seq_len = 15
model_name = 'transformer'

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # data directories
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()

def save_history(path, history):
    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
               history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4) 

def get_train_data(training):
#     train_data = pd.read_csv(training)
    train_data = pd.read_csv(os.path.join(training, 'btc_train.csv'))
    # Training data    
    train_data = train_data.values    
    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 240 df-rows
        y_train.append(train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row 240+1
    X_train, y_train = np.array(X_train), np.array(y_train)
    print('x train', X_train.shape,'y train', y_train.shape)    
    return X_train, y_train

def get_val_data(validation):
#     val_data = pd.read_csv(validation)
    val_data = pd.read_csv(os.path.join(validation, 'btc_val.csv'))
    # Validation data    
    val_data = val_data.values
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i-seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)
    print('x val', X_val.shape,'y val', y_val.shape)
    return X_val, y_val

def get_model(learning_rate):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    with mirrored_strategy.scope():
        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(seq_len)
        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        
        '''Construct model'''
        in_seq = tf.keras.Input(shape=(seq_len, 6))
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
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mape', 'accuracy'])
    print(model.summary())    
    return model 


if __name__ == '__main__':
    
    args, _ = parse_args()
    print('---------------  args ------------------')
    print(args)
    
    x_train, y_train = get_train_data(args.training)
    x_val, y_val = get_val_data(args.validation)  
    
    model = get_model(args.learning_rate)
    
    history = model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_val, y_val))
    print('---------------  saving model ------------------')
    save_history(args.model_dir + "/history.p", history)
    
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    model.save(args.model_dir + '/1')
    print('---------------  DONE --------------------------')
    print('Model saved to {}'.format(args.model_dir))

