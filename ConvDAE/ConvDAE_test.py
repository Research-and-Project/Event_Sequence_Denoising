
# -*- coding: utf-8 -*-
'''
卷积自编码器
累积重构-卷积去噪自编码ConvDAE_test
[0] ConvDAE_3
[1] ConvDAE_semisup:
+：半监督:输入MINST_pic,dropout; 真值MNIST_pic_gt
'''
# In[45]:
#导入基本模块
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import os
from my_tf_lib import my_io

# In[9]:
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# In[9]:
#重置tensorboard graph
tf.reset_default_graph() 

# In[]
# 路径参数设置
# 参数
epochs = 200
batch_size = 128
learning_rate=0.001

# 标志
isTrain = True

# 路径
path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
# path3 = "/data/zzh/dataset/MNIST_diff/MNIST_diff_train.mat" #diff

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "ConvDAE_semisup--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/ConvDAE_semisup_'+timestamp
test_log_dir = 'logs/test/ConvDAE_semisup_'+timestamp


# In[]:
# 加载数据
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)
# diff_data = my_io.load_mat(path3)

train_x = train_data['N_MNIST_pic_train'].astype('float32')
train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
# train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
# train_y = diff_data['train_diff_gt'].astype('float32') #gt使用diff_gt

test_x = test_data['N_MNIST_pic_test'].astype('float32')
test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

# 数据打印测试
# for k in range(5):
#     plt.subplot(2,5,k+1)
#     plt.imshow(train_x[k])
#     plt.title('train_x_%d'%(k+1))
#     plt.xticks([])
#     plt.yticks([])        
#     plt.subplot(2,5,k+6)
#     plt.imshow(train_y[k])
#     plt.title('train_y_%d'%(k+1))
#     plt.xticks([])
#     plt.yticks([])


# In[]:
# 构造模型
# 输入
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #弃权概率0.0-1.0  1.0表示不使用弃权

#选择激活函数
# act_fun = tf.nn.relu 
act_fun = tf.nn.tanh
act_fun_out = tf.nn.tanh

# Encoder
with tf.name_scope('encoder'):
    drop = tf.nn.dropout(inputs_, keep_prob)  # dropout
    conv1 = tf.layers.conv2d(drop, 64, (3,3), padding='same', activation=act_fun)
    # conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=act_fun)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
    # conv1 = tf.nn.dropout(conv1, keep_prob)

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=act_fun)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=act_fun)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
    # conv3 = tf.nn.dropout(conv3, keep_prob)

# Decoder
with tf.name_scope('decoder'):
    conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=act_fun)
    # conv4 = tf.nn.dropout(conv4, keep_prob)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=act_fun)
    # conv5 = tf.nn.dropout(conv5, keep_prob)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=act_fun)
    # conv6 = tf.nn.dropout(conv6, keep_prob)

# logits and outputs
with tf.name_scope('outputs'):
    logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

    #dropout层
    # logits_ = tf.nn.dropout(logits_,keep_prob)

    # outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
    outputs_ = act_fun_out(logits_, name='outputs_')

# loss and Optimizer
with tf.name_scope('loss'):
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
#     loss = tf.reduce_sum(tf.square(targets_ -  outputs_))

    loss = tf.losses.mean_squared_error(targets_ , outputs_)

    cost = tf.reduce_mean(loss)
    tf.summary.scalar('cost', cost)
    
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[]
# 初始化会话和模型保存器及tensorboard
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# In[]:
# 训练
if isTrain:
    for e in range(epochs):
        for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y):
            x = batch_x.reshape((-1, 28, 28, 1))
            y = batch_y.reshape((-1, 28, 28, 1))
            sess.run(optimizer, feed_dict={inputs_: x, targets_: y,keep_prob:0.7})

        if e%5 == 0:
            test_x1 = test_x[500:10000:1000].reshape((-1, 28, 28, 1))
            test_y1 = test_y[500:10000:1000].reshape((-1, 28, 28, 1))  
            tr, tr_cost = sess.run([merged, cost], feed_dict={inputs_: x, targets_: y,keep_prob:0.7})
            te, te_cost = sess.run([merged, cost], feed_dict={inputs_: test_x1, targets_: test_y1,keep_prob:1.0})
        
            writer_tr.add_summary(tr, e)
            writer_te.add_summary(te, e)     

            print(e,"Train cost:",tr_cost,"Test cost",te_cost)

        
        if e%20 == 0 and e!=0:
            saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
            # saver.save(sess,model_path+'my_model') 
            print('epoch %d model saved to:'%e, model_path+'my_model')
        
    saver.save(sess,model_path+'my_model') 
    print('epoch: %d model saved to:'%e, model_path+'my_model') 


# In[]:
# 测试
if not isTrain:
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))

    in_imgs = test_x[500:10000:1000]
    gt_imgs = test_y[500:10000:1000]

    reconstructed = sess.run(outputs_, 
                            feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})

    # zzh:三值化，阈值-0.5，0.5        
    # reconstructed[reconstructed<=-0.5] = -1.
    # reconstructed[reconstructed>=0.5] = 1.
    # reconstructed[(reconstructed>-0.5) & (reconstructed<0.5)] = 0.

    for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((28, 28)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)


# In[ ]:
 #终止化
sess.close()
writer_tr.close()
writer_te.close()





