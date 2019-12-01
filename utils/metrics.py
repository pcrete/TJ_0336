import numpy as np
import tensorflow as tf
import pandas as pd



df = pd.read_csv('./data/005_weights.csv')
weight = tf.convert_to_tensor(df['weight'].values.reshape((1, -1)))

class Metric:
    
    def __init__(self, weights):
        self.W = weights

    def eval_metric(self, y_true, y_pred):
        loss = 0

        for i in range(len(y_pred)):

            ind = np.argmax(y_true[i])

            y_ = y_pred[0][ind]
            ln = np.log(y_)

            loss += self.W[ind]*ln

        return (-1*loss)/len(y_pred)
    
def loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))