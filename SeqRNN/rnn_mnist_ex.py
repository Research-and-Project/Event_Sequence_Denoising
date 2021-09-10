# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:01:43 2019

RNN net MNIST recognition

@author: dawnlh
"""
# In[]

#导入数据
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected #全连接层
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os

# In[]

#reset graph to avoid error
tf.reset_default_graph()  


# In[]

#参数设置
n_steps=28 #512   #步长
n_inputs=28 #3 #输入数据个数(特征维度)
n_neurons=128 #每层神经元的数量
n_outputs=10  #输出数据（
n_epochs=50
batch_size=128
learning_rate=0.001

# In[ ]:

# 加载数据

mnist = input_data.read_data_sets("D:/1-Document/毕业设计/zzh/zzh-工程/data/MNIST_data")
X_test=mnist.test.images.reshape((-1,n_steps,n_inputs))  #转换成n个28x28的测试集
y_test=mnist.test.labels

# In[]

# 网络构建
with tf.name_scope('inputs'):
    X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])#输入32步和32个X输入
    y=tf.placeholder(tf.int32,[None])

#he_init=tf.contrib.layers.variance_scaling_initializer()#He initialization 参数初始化
#with tf.variable_scope("rnn",initializer=he_init):】
with tf.name_scope('RNN'):    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits=fully_connected(states,n_outputs,activation_fn=None)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)

with tf.name_scope('evaluation'):
    loss=tf.reduce_mean(xentropy)
    tf.summary.scalar('loss', loss)
#    tf.summary.scalar('test_loss', loss)
    
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))     
    tf.summary.scalar('accuracy', accuracy)
#    tf.summary.scalar('test_accuracy', accuracy)
    
with tf.name_scope('train'):        
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

init = tf.global_variables_initializer()

# tensorboard
merged = tf.summary.merge_all() 

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
train_log_dir = 'logs/train/RNN_MNIST_'+timestamp
test_log_dir = 'logs/test/RNN_MNIST_'+timestamp


# In[]

# 网络训练
with tf.Session() as sess:
    
    init.run()
    
    writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
    writer_te = tf.summary.FileWriter(test_log_dir)
    
    num_batch = int(mnist.train.num_examples/batch_size)
    
    for epoch in range(n_epochs):
        for iteration in range(num_batch):
            x_batch,y_batch=mnist.train.next_batch(batch_size)
            x_batch=x_batch.reshape((-1,n_steps,n_inputs))#转换成batch_size个28x28的输入
            sess.run(training_op,feed_dict={X:x_batch,y:y_batch}) 
            
        if epoch%2 == 0:
            acc_train=accuracy.eval(feed_dict={X:x_batch,y:y_batch})
            acc_test=accuracy.eval(feed_dict={X: X_test, y: y_test})           
            print(epoch,"train accuracy:",acc_train,"Test accuracy",acc_test)
            te = sess.run(merged,feed_dict={X:X_test,y:y_test})
            writer_te.add_summary(te, epoch)  
            
        else:
            tr = sess.run(merged,feed_dict={X:x_batch,y:y_batch})                
            writer_tr.add_summary(tr, epoch)
    
    # 模型保存
    root_path = "D:/1-Document/data/model_data/MNIST/RNN_MNIST/"
    model_dir = "RNN_MNIST--" +  timestamp
    model_path = root_path + model_dir
    
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    saver = tf.train.Saver()
    saver.save(sess,model_path+'my_model')   
    print('model saved to:', model_path+'my_model')            
 
# In[]
# 测试代码 
#mnist = input_data.read_data_sets(r"D:/1-Document/毕业设计/zzh/zzh-工程/data/MNIST_data")
#X_test=mnist.test.images.reshape((-1,n_steps,n_inputs))  #转换成n个28x28的测试集
#y_test=mnist.test.labels
#x_batch,y_batch=mnist.train.next_batch(batch_size)           
#
##print('mnist:', mnist.shape)
#print('x_batch:', x_batch.shape, '\ty_batch:', y_batch.shape) 
#print('X_test:', X_test.shape, '\ty_test:', y_test.shape) 
            