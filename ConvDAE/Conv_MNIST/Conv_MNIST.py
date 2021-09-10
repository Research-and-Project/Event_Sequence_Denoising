# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:36:05 2019

Convnet MNIST recognition

@author: dawnlh
"""

# In[]
# 导入模块
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from datetime import datetime
import os

# In[]
#图设置
tf.reset_default_graph()  


# In[]
# 导入数据
mnist = input_data.read_data_sets("D:/1-Document/毕业设计/zzh/zzh-工程/data/MNIST_data") #MNIST数据集所在路径


# In[]
# 函数定义
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# In[]
#参数定义
#超参数
lr = 1e-4
n_epoch = 20
batch_size = 128

# 网络参数定义
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# In[]
# 网络结构
x = tf.placeholder(tf.float32, [None, 28, 28,1])
y_ = tf.placeholder(tf.int64, [None,])
keep_prob = tf.placeholder("float")

# 第一层：conv
h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层：conv
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层：reshape + fc
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 第三层：dropout + fc
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_out = tf.nn.softmax(logits)

pred = tf.argmax(y_out,1)

# loss function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_out))

# optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_out,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    num_batch = int(mnist.train.num_examples/batch_size)
    
    for i in range(n_epoch):
        for iteration in range(num_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            
            feed_dict={x: np.reshape(x_batch,[-1,28,28,1]), y_: y_batch, keep_prob: 0.8} #zzh
            
            train_step.run(feed_dict=feed_dict)
        if i%2 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            test_accuracy = accuracy.eval(feed_dict={x:  np.reshape(mnist.test.images,[-1,28,28,1]), y_: mnist.test.labels, keep_prob: 1.0})
            print('step %d, train accuracy %g, test accuracy %g ' %(i, train_accuracy, test_accuracy))
    
    # 模型保存
    timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
    root_path = "D:/1-Document/data/model_data/MNIST/Conv_MNIST/"
    model_dir = "Conv_MNIST--" +  timestamp
    model_path = root_path + model_dir
    
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    saver = tf.train.Saver()
    saver.save(sess,model_path+'my_model')   
    print('model saved to:', model_path+'my_model')