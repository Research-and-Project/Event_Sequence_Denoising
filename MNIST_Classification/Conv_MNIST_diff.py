# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:36:05 2019

Convnet MNIST recognition

使用说明：
使用diff数据集时修改内容：三个路径名， 一个batch_iter， 一个feed_dict2
@author: dawnlh
"""

# In[]
# 导入模块
import tensorflow as tf
import numpy as np
from datetime import datetime
from my_tf_lib import my_io
import os

# In[]
#图设置
tf.reset_default_graph()  



# In[]:
# 路径参数

# 路径
#pic_test_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
#pic_train_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
diff_train_data_path = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_train.mat"
diff_test_data_path = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_test.mat"
diff_label_data_path = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_labels.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())

model_root_path = "D:/1-Document/data/model_data/MNIST/Conv_MNIST_diff/"
model_dir = "Conv_MNIST_diff_y--" +  timestamp
model_path = model_root_path + model_dir
if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/Conv_MNIST_diff_y_'+timestamp
test_log_dir = 'logs/test/Conv_MNIST_diff_y_'+timestamp

#参数
lr = 1e-4
n_epoch = 50
batch_size = 128

# In[]
# 导入数据
#mnist = input_data.read_data_sets("D:/1-Document/毕业设计/zzh/zzh-工程/data/MNIST_data") #MNIST数据集所在路径

#pic_train_data = my_io.load_mat(pic_train_data_path)
#pic_train_x = pic_train_data['N_MNIST_pic_train'].astype('float32')
#pic_train_y = pic_train_data['N_MNIST_pic_train_gt'].astype('float32')
#print('pic_train_x: ', pic_train_x.shape, '\tpic_train_y: ', pic_train_y.shape)
#
#pic_test_data = my_io.load_mat(pic_test_data_path)
#pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
#pic_test_y = pic_test_data['N_MNIST_pic_test_gt'].astype('float32')
#print('pic_test_x: ', pic_test_x.shape, '\tpic_test_y: ', pic_test_y.shape)
#
diff_test_data = my_io.load_mat(diff_test_data_path)
diff_train_data = my_io.load_mat(diff_train_data_path)
diff_label_data = my_io.load_mat(diff_label_data_path)

diff_test_x = diff_test_data['test_diff'].astype('float32')
diff_test_y = diff_test_data['test_diff_gt'].astype('float32')
diff_test_labels_onehot = diff_label_data['test_labels'].astype('int64')
diff_test_labels = np.array([np.argmax(diff_test_labels_onehot)for diff_test_labels_onehot in diff_test_labels_onehot]) #独热编码转换为普通编码

diff_train_x = diff_train_data['train_diff'].astype('float32')
diff_train_y = diff_train_data['train_diff_gt'].astype('float32')
diff_train_labels_onehot = diff_label_data['train_labels'].astype('int64')
diff_train_labels = np.array([np.argmax(diff_train_labels_onehot)for diff_train_labels_onehot in diff_train_labels_onehot]) #独热编码转换为普通编码

print('diff_test_x: ', diff_test_x.shape, '\tdiff_test_y: ', diff_test_y.shape, '\tdiff_test_labels: ',diff_test_labels.shape)
print('diff_train_x: ', diff_train_x.shape, '\tdiff_train_y: ', diff_train_y.shape, '\tdiff_train_labels: ',diff_train_labels.shape)

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


# optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_out,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)
  
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    # tensorboard
    merged = tf.summary.merge_all() 
    writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
    writer_te = tf.summary.FileWriter(test_log_dir)    
    
    for i in range(n_epoch):
        for x_batch, y_batch in my_io.batch_iter(batch_size, diff_train_y, diff_train_labels):   #zzh:change dataset here
            
            feed_dict1={x: np.reshape(x_batch,[-1,28,28,1]), y_: y_batch, keep_prob: 0.8} 
            
            sess.run(train_step, feed_dict=feed_dict1)
        if i%2 == 0:
            train_accuracy, tr = sess.run([accuracy, merged], feed_dict=feed_dict1)
            
            feed_dict2={x: np.reshape(diff_test_y,[-1,28,28,1]), y_: diff_test_labels, keep_prob: 1.0}  #zzh:change dataset here
            
            test_accuracy, te= sess.run([accuracy, merged],feed_dict=feed_dict2)
            print('step %d, train accuracy %g, test accuracy %g ' %(i, train_accuracy, test_accuracy))           
            writer_tr.add_summary(tr, i)            
            writer_te.add_summary(te, i) 
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess,model_path+'my_model')   
    print('model saved to:', model_path+'my_model')    