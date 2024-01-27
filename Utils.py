
import numpy as np

def batch(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]
    

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

