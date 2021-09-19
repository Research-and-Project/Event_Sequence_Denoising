# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:46:04 2019
模型加载与预测 2
for example data

+： 对x_data的时间维度进行归一化
@author: dawnlh
"""

# In[]

#导入模块
import tensorflow as tf
import numpy as np
from my_evt_lib import my_cvt as mc
from my_tf_lib import my_io
from time import sleep
import os

# In[]
#图设置
tf.reset_default_graph()  

# In[]
# 路径参数
#test_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/full scale/N_MNIST_seq_test.mat"
#test_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/sample/N_MNIST_seq_test-1024.mat"
test_data_path = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/sample/example.mat"

root_model_path = "D:/1-Document/data/model_data/SeqRNN/good_bak/"
#root_model_path = "D:/1-Document/data/model_data/SeqRNN/"
model_dir = "SeqRNN_5--04-07_14-07(acc-0.768)/"
model_path = root_model_path + model_dir
model_name = 'my_model'
model_ind = 2
time_std = 2 # 时间归一化标志：0-步归一化；1-批归一化

n_dataseg = 204800 # example test data
#n_dataseg = 1024 # N_MNIST test data
n_steps=64   #步长
#n_steps=128   #SeqRNN_2_1步长
n_inputs=4 #输入数据个数(特征维度)
n_neurons=32 #每层神经元的数量
n_outputs=3  #输出数据（三种输出分别代表-1,0,1, 此处实际输出为类别序号，分别为0,1,2）

# In[]

#模型加载
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restorer = tf.train.import_meta_graph(model_path+model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("inputs/Placeholder:0")
y = graph.get_tensor_by_name("inputs/Placeholder_1:0")

outputs = graph.get_tensor_by_name("outputs/Reshape:0")
accuracy = graph.get_tensor_by_name("evaluation/Mean_1:0")

# In[ ]:

#数据加载
test_data = my_io.load_mat(test_data_path)
test_x = test_data['example'].astype('float32')
test_y = test_data['example_gt'].astype('float32')

#test_x = test_data['N_MNIST_seq_test'].astype('float32')
#test_y = test_data['N_MNIST_seq_test_gt'].astype('float32')

print('test_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

#test_x = test_data['evtmat'].astype('float32')

# 数据测试
#for i in range(1):
##     i = i * 9000 + 900 
#     evt_ar1 = test_x[i,:,:]
#     evt_ar2 = test_y[i,:,:] 
#    
#     evt_td1 = mc.array2evt(evt_ar1, 600, 480) 
#     evt_td2 = mc.array2evt(evt_ar2, 600, 480)  
#      
#     evt_td1.show_td(10,0.01)
#     evt_td2.show_td(10,0.02)

# In[ ]:
#数据归一化
x_max_test = np.max(test_x[:,0,:])
y_max_test = np.max(test_x[:,1,:])
t_max_test = np.max(test_x[:,2,:])

test_x[:,0,:] = test_x[:,0,:]/x_max_test
test_x[:,1,:] = test_x[:,1,:]/y_max_test
test_x[:,2,:] = test_x[:,2,:]/t_max_test

test_y[:,0,:] = test_y[:,0,:]/x_max_test
test_y[:,1,:] = test_y[:,1,:]/y_max_test
test_y[:,2,:] = test_y[:,2,:]/t_max_test

# 数据极性化表示
test_x[:,3,:] = test_x[:,3,:]+1 #将-1,0,1的极性表示转换为0,1,2以适应cross_entropyloss
X_test = test_x.transpose(0,2,1) # X变为 【样本数 ， n_step， 特征维度】
test_y[:,3,:] = test_y[:,3,:]+1
y_test = test_y[:,3,:]

# 修改步长
X_test = X_test.reshape((-1,n_steps,n_inputs))
y_test = y_test.reshape((-1,n_steps))


print('X_test: ',X_test.shape, '\ty_test: ', y_test.shape)

y_test1 = y_test.flatten()   #测试时输入标签
# In[] 预测
# 预测
x_data = np.copy(X_test)
ydata = np.copy(y_test)

x_data_ =  np.copy(X_test)

if time_std == 0:
    x_data_[:,:,2] = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in x_data_[:,:,2]])  #对时间维度进行“步归一化”
if time_std == 1:
    x_data_[:,:,2] = np.array((x_data_[:,:,2] - np.min(x_data_[:,:,2]))/(np.max(x_data_[:,:,2]) -np.min(x_data_[:,:,2]))) #对时间维度进行“批归一化”

#ypreds_= sess.run(outputs, feed_dict={X: x_data_}) #for SeqRNN0,1,3--RNN
acc, ypreds_= sess.run([accuracy, outputs], feed_dict={X: x_data_, y:y_test1})  #for SeqRNN2--LSTM
print('test accuracy: ', acc)

preds_data =  np.copy(x_data)  #注意复制时要deepcopy才行
preds_data[:,:,3] = ypreds_

gt_data =  np.copy(X_test)
gt_data[:,:,3] = ydata

# zzh:修改步长：原数据的长度是512，训练时截断为 n_steps
x_data = x_data.reshape((-1,n_dataseg,n_inputs))
preds_data = preds_data.reshape((-1,n_dataseg,n_inputs))
gt_data = gt_data.reshape((-1,n_dataseg,n_inputs))

# 转换为【样本数*特征维数*样本长度】的形式
x_data = x_data.transpose(0,2,1)
preds_data =preds_data.transpose(0,2,1)
gt_data = gt_data.transpose(0,2,1)

# 去归一化
x_data[:,0,:] = x_data[:,0,:]*x_max_test
x_data[:,1,:] = x_data[:,1,:]*y_max_test
x_data[:,2,:] = x_data[:,2,:]*t_max_test

preds_data[:,0,:] = preds_data[:,0,:]*x_max_test
preds_data[:,1,:] = preds_data[:,1,:]*y_max_test
preds_data[:,2,:] = preds_data[:,2,:]*t_max_test

gt_data[:,0,:] = gt_data[:,0,:]*x_max_test
gt_data[:,1,:] = gt_data[:,1,:]*y_max_test
gt_data[:,2,:] = gt_data[:,2,:]*t_max_test


preds_data[:,3,:] = preds_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式
x_data[:,3,:] = x_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式
gt_data[:,3,:] = gt_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式

# In[]
# 显示
for i in range(1):
#    x_data_i = x_data[i,...]     
    preds_data_i = preds_data[i,...]
#    gt_data_i = gt_data[i,...]
    
#    evt_data = mc.array2evt(x_data_i, 600, 480)    
    evt_preds = mc.array2evt(preds_data_i, 600, 480)
#    evt_data_gt = mc.array2evt(gt_data_i, 600, 480)
    
#    vid = evt_data.show_td(10, 0.005, return_pic_flag=0)
    vid_preds = evt_preds.show_td(20,0.005, return_pic_flag=0)
#    vid_gt =  evt_data_gt.show_td(20,0.005, return_pic_flag=0)
    
# In[]

# 终止化

#sess.close()
    
# In[]
# 视频显示
wait_delay = 10 
import cv2
for frame in vid_preds:
    cv2.imshow('img', frame)
    cv2.waitKey(wait_delay)
cv2.destroyAllWindows()
