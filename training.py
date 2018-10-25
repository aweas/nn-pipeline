import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import pylab
import matplotlib.pyplot as plt
from network import AbstractNetwork
from dataset_explorer import DatasetExplorer
import logging
from tensorflow.nn import softmax_cross_entropy_with_logits
from sklearn import preprocessing

batch_size = 128
epoch_num = 10
input_shape = (64, 256, 1)

def get_data():
    explorer = DatasetExplorer(r"Q:\Wydymanski Witold\Tramwaje\Aria\records\session_new.csv")
    explorer.prepare()
    X, y = explorer.get_data()
    return X, y

def clean_data(X, y):
    print('Proportions before clearing')
    for i in list(set(y)):
        l = len(y[y==i])
        print(i, l)

    y_retain = []
    num_samples = len(y[y==y[0]])
    for i in list(set(y)):
        if l>0.1*len(y):
            y_retain.append(i)
            num_samples = min(num_samples, l)
    
    print('\nNumber of remaining classes:', len(y_retain))
    print('Number of samples per class:', num_samples)

    X_new = []
    y_new = []
    for y_item in y_retain:
        X_new.extend(X[y==y_item][:num_samples])
        y_new.extend(y[y==y_item][:num_samples])

    l = len(y[y==y[0]])

    classes_num = len(set(y))
    samples_num = int(len(y)/4)*4

    X = np.asarray(X_new)
    y = np.asarray(y_new)

    X = X[:samples_num]
    y = y[:samples_num]

    print(f'Classes num: {classes_num}')
    print(f'Samples num: {samples_num}')
            
    print('\nNumber of remaining classes:', len(set(y)))
    print('Samples per class:', l)
    return X, y

X, y = clean_data(*get_data())
classes_num = len(list(set(y)))

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

class CnnNetwork(AbstractNetwork):       
    def _inference(self):
        inp = self.input
        with tf.name_scope('classificator'):                
            layer = tf.layers.conv2d(inp, 4, 3, activation=tf.nn.relu)
            layer = tf.layers.conv2d(layer, 8, 3, activation=tf.nn.relu)
            layer = tf.layers.conv2d(layer, 8, 3, activation=tf.nn.relu)
            layer = tf.layers.conv2d(layer, 8, 3, activation=tf.nn.relu)
            layer = tf.layers.conv2d(layer, 8, 3, activation=tf.nn.relu)
            layer = tf.layers.conv2d(layer, classes_num, 3, activation=tf.nn.relu)

            layer = tf.keras.layers.GlobalAveragePooling2D()(layer)
            layer = tf.nn.softmax(layer)
        return layer

network = CnnNetwork(input_shape, classes_num=classes_num)
network.set_training_data(X, y)
network.training(batch_size=4, epochs=5, iter_before_validation=1)