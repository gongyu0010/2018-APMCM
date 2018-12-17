import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), 'W')
            tf.summary.histogram(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal([1, out_size]), name='b')
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

            if activation_function is None:
                outputs = Wx_plus_b
                tf.summary.histogram(layer_name + '/outputs', outputs)
            else:
                outputs = activation_function(Wx_plus_b)
                tf.summary.histogram(layer_name + '/outputs', outputs)
            return outputs


with tf.name_scope('inputs'):
    x_data = tf.placeholder(shape=[76,25], dtype=tf.float32, name='x-input')
    y_data = tf.placeholder(shape=[76,1], dtype=tf.float32, name='y-input')

l1 = add_layer(x_data, 25, 15, layer_name='layer_01', activation_function=tf.nn.relu)
prediction = add_layer(l1, 15, 1, layer_name='layer_02', activation_function=tf.nn.sigmoid)


with tf.name_scope('loss'):
    loss = -tf.reduce_mean(prediction * tf.log(tf.clip_by_value(y_data, 1e-10, 1.0)))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

feature = pd.read_csv('feature.csv')
feature = np.array(feature)
feature = preprocessing.scale(feature)
print("-----------------------------------------------------------------------------------------------------------------")
print("                                            feature loaded                                                       ")
print("-----------------------------------------------------------------------------------------------------------------")
target = pd.read_csv("target.csv")
target = np.array(target)
print("-----------------------------------------------------------------------------------------------------------------")
print("                                            target loaded                                                       ")
print("-----------------------------------------------------------------------------------------------------------------")

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)

    STEPS = 5000
    for i in range(STEPS):
        sess.run(train_step, feed_dict={x_data:feature[:,:], y_data:target[:,:]})
        if i % 100 is 0:
            loss_total = sess.run(loss, feed_dict={x_data:feature[:,:], y_data:target[:,:]})
            print("After %d training steps, loss on all data is %g" % (i, loss_total))
            result = sess.run(merged, feed_dict={x_data:feature[:,:], y_data:target[:,:]})
            writer.add_summary(result, i)