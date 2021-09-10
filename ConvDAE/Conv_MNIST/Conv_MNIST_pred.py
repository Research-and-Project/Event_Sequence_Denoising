# -*- coding: utf-8 -*-
"""
Created on Wed April 2 10:00:00 2019

Convnet MNIST recognition 预测
使用说明：
修改内容：导入数据集标志in_data_flag 和相应项下面的测试数据testX
@author: dawnlh
"""

# In[]
# 导入模块
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from my_tf_lib import my_io
from my_imgproc_lib import my_img_evaluation as my_evl
from datetime import datetime
import os


# In[]
#图设置
tf.reset_default_graph()  

# In[]:
# 路径参数
#pic_train_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
pic_test_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
pic_label_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_labels.mat"

diff_test_data_path = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_test.mat"
diff_label_data_path = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_labels.mat"

root_model_path = "D:/1-Document/data/model_data/MNIST/Conv_MNIST/"
#root_model_path = "D:/1-Document/data/model_data/MNIST/Conv_MNIST_diff/"
#root_model_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "Conv_MNIST--04-02_11-02/"
model_path = root_model_path + model_dir
model_name = 'my_model'

reconstruct_data_dir = 'ConvDAE_2--03-29_10-36(cost-0.027)/'
pred_res_path = 'D:/1-Document/毕业设计/zzh/zzh-工程/ConvDAE/predict_res/'+reconstruct_data_dir

batch_size = 128

# In[]
# 导入数据
in_data_flag = 4

if in_data_flag == 1: # mnist数据
    mnist = input_data.read_data_sets("D:/1-Document/毕业设计/zzh/zzh-工程/data/MNIST_data") #MNIST数据集所在路径
    testX = np.reshape(mnist.test.images, (-1, 28, 28, 1)) 
    testY = mnist.test.labels

if in_data_flag == 2: # 算法重建数据       
    reconstruct = np.load(pred_res_path+'pred_res.npy')
    pic_label_data = my_io.load_mat(pic_label_data_path) #注意这个数据集的labels不是独热编码，是普通编码，但是是二维的，需要squeeze一下
    
    reconstruct = abs(reconstruct) #取绝对值    
    pic_test_labels = np.squeeze(pic_label_data['test_labels'].astype('int64')) #变成一维
    pic_test_labels = pic_test_labels[0:10000:10] #zzh：修改项--导入的重建数据不是全部的测试集重建数据
    
    print('pic_reconstruct: ', reconstruct.shape, '\tpic_test_labels: ',pic_test_labels.shape)    

    testX = np.reshape(reconstruct, (-1, 28, 28, 1)) #zzh:修改
    testY = pic_test_labels    

if in_data_flag == 3:    # N_MNIST_pic_test 数据  
    pic_test_data = my_io.load_mat(pic_test_data_path)
    pic_label_data = my_io.load_mat(pic_label_data_path)
    
    pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
    pic_test_y = pic_test_data['N_MNIST_pic_test_gt'].astype('float32')
    
    # 数据预处理
    #pic_test_x = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in pic_test_x]) #归一化像素值到【0-1】
    #pic_test_y = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in pic_test_y])   
    pic_test_x = abs(pic_test_x) #取绝对值
    pic_test_y = abs(pic_test_y)
    
    pic_test_labels = np.squeeze(pic_label_data['test_labels'].astype('int64')) #变成一维
    
    print('pic_test_x: ', pic_test_x.shape, '\tpic_test_y: ', pic_test_y.shape, '\tpic_test_labels: ',pic_test_labels.shape)
    
    testX = np.reshape(pic_test_y, (-1, 28, 28, 1)) #zzh:修改
    testY = pic_test_labels
   
if in_data_flag == 4:   # N_MNIST_test 数据    
    diff_test_data = my_io.load_mat(diff_test_data_path)
    diff_label_data = my_io.load_mat(diff_label_data_path)
    
    diff_test_x = diff_test_data['test_diff'].astype('float32')
    diff_test_y = diff_test_data['test_diff_gt'].astype('float32')
    
    # 数据预处理
    #diff_test_x = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in diff_test_x]) #归一化像素值到【0-1】
    #diff_test_y = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in diff_test_y])   
    diff_test_x = abs(diff_test_x) #取绝对值
    diff_test_y = abs(diff_test_y)
    
    diff_test_labels_onehot = diff_label_data['test_labels'].astype('int64')
    diff_test_labels = np.array([np.argmax(diff_test_labels_onehot)for diff_test_labels_onehot in diff_test_labels_onehot]) #独热编码转换为普通编码
    print('diff_test_x: ', diff_test_x.shape, '\tdiff_test_y: ', diff_test_y.shape, '\tdiff_test_labels: ',diff_test_labels.shape)
    
    testX = np.reshape(diff_test_y, (-1, 28, 28, 1)) #zzh:修改
    testY = diff_test_labels

#if in_data_flag = 4:    
    #pic_train_data = my_io.load_mat(pic_train_data_path)
    #pic_train_x = pic_train_data['N_MNIST_pic_train'].astype('float32')
    #pic_train_y = pic_train_data['N_MNIST_pic_train_gt'].astype('float32')
    #print('pic_train_x: ', pic_train_x.shape, '\tpic_train_y: ', pic_train_y.shape)
    #
    #pic_test_data = my_io.load_mat(pic_test_data_path)
    #pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
    #pic_test_y = pic_test_data['N_MNIST_pic_test_gt'].astype('float32')
    #print('pic_test_x: ', pic_test_x.shape, '\tpic_test_y: ', pic_test_y.shape)
    
    #testX = np.reshape(reconstruct, (-1, 28, 28, 1))
    #testY = pic_test_labels[500:10000:100] 
# In[]:
# 数据打印测试
#for k in range(5):
#    plt.subplot(2,5,k+1)
#    plt.imshow(test_x[k])
#    plt.title('test_x_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])        
#    plt.subplot(2,5,k+6)
#    plt.imshow(test_y[k])
#    plt.title('test_y_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])

# In[]

#模型加载
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restorer = tf.train.import_meta_graph(model_path+model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[0])

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
y_ = graph.get_tensor_by_name("Placeholder_1:0")
keep_prob = graph.get_tensor_by_name("Placeholder_2:0")
pred = graph.get_tensor_by_name("ArgMax:0")
accuracy = graph.get_tensor_by_name("Mean:0")


# In[]:
# 预测
test_accuracy = sess.run(accuracy,feed_dict={x: testX, y_: testY, keep_prob: 1.0})
preds = sess.run(pred,feed_dict={x: testX, y_: testY, keep_prob: 1.0})
print('current accuracy:', test_accuracy)

in_imgs = testX[0:10000:1000]
labels = testY[0:10000:1000]
predicts = preds[0:10000:1000]

fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(20,4))

for images, ax, label, predict in zip(in_imgs, axes, labels, predicts):
        ax.imshow(images.reshape((28, 28)))
        ax.set_title('label:%d; pred:%d'%(label, predict),fontsize=12,color='r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)

#for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
#    for img, ax in zip(images, row):
#        ax.imshow(img.reshape((28, 28)))
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#fig.tight_layout(pad=0.1)


# In[24]:
#sess.close()

