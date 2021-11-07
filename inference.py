import json
import numpy as np
import pandas as pd
import logging

seq_len = 240
def get_test_data(test):
    test_data = pd.read_csv(test)
    test_data = test_data.values
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i-seq_len:i])
        y_test.append(test_data[:, 3][i])
    X_test, y_test = np.array(X_test), np.array(y_test)
    logging.info('x test', X_test.shape,'y test', y_test.shape)
    return X_test, y_test


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/npy':
        return json.dumps({"instances": data.read()})
    
    if context.request_content_type == 'text/csv':
        csv_file = data.read().decode('utf-8')
        csv_data_test_x, csv_data_test_y = get_test_data(csv_file)
        # very simple csv handler
        return json.dumps({
            'instances': csv_data_test_x
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type

