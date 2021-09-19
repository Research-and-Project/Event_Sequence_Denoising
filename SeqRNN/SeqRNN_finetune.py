# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:06:50 2019

# SeqRNN 断点续训 finetune

+： 对X_test和x_batch的时间维度进行归一化
@author: dawnlh
"""

# In[]
#导入模块
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected #全连接层
from event_python import eventvision as ev
from my_evt_lib import my_cvt as mc
from my_tf_lib import my_io
from time import sleep, time
from datetime import datetime
import os


# In[]
#运行环境配置
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


# In[]
#reset graph to avoid error
tf.reset_default_graph()  


# In[]
#参数设置
n_steps=64   #步长
n_inputs=4 #输入数据个数(特征维度)
#n_neurons=32 #每层神经元的数量
n_neurons = [64, 32] 
n_outputs=3  #输出数据（三种输出分别代表-1,0,1, 此处实际输出为类别序号，分别为0,1,2）
n_epochs=100
batch_size=128
learning_rate=0.001
model_index = -1

# 标志
is_Finetune = True #微调模型标志
time_std = 1;  # 时间归一化标志：0-步归一化；1-批归一化

t1 = time()

# 路径
path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/full scale/N_MNIST_seq_train.mat"
path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/full scale/N_MNIST_seq_test.mat"
#path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/long step/N_MNIST_seq_train.mat"
#path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/long step/N_MNIST_seq_test.mat"
#path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/sample/example.mat" # example只是一个测试集，这里当成训练集用
#path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/sample/example.mat" # 测试集与训练集相同

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
train_log_dir = 'logs/train/SeqRNN_4T_'+timestamp
test_log_dir = 'logs/test/SeqRNN_4T_'+timestamp
root_path = "D:/1-Document/data/model_data/SeqRNN/"
model_dir = "SeqRNN_4T_--" +  timestamp
model_path = root_path + model_dir
model_name = 'my_model'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
    
# old model to fnetune
if is_Finetune: 
#    root_old_model_path = "D:/1-Document/data/model_data/SeqRNN/good_bak/"
    root_old_model_path = "D:/1-Document/data/model_data/SeqRNN/"
    old_model_dir = "SeqRNN_4T_--04-11_22-41/"
    old_model_path = root_old_model_path + old_model_dir
    old_model_name = 'my_model'
# In[ ]:
# 加载数据
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

train_x = train_data['N_MNIST_seq_train'].astype('float32')
train_y = train_data['N_MNIST_seq_train_gt'].astype('float32')
test_x = test_data['N_MNIST_seq_test'].astype('float32')
test_y = test_data['N_MNIST_seq_test_gt'].astype('float32')

#example data
#train_x = train_data['example'].astype('float32')
#train_y = train_data['example_gt'].astype('float32') 
#test_x = test_data['example'].astype('float32')
#test_y = test_data['example_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape)
print('test_x: ', test_x.shape, '\ttest_y: ', test_y.shape)


# In[ ]:
# 数据测试
#for i in range(5):
#     i = i * 9000 + 900 
#     evt_ar1 = train_x[i,:,:]
#     evt_ar2 = train_y[i,:,:] 
#    
#     evt_td1 = mc.array2evt(evt_ar1)
#     evt_td2 = mc.array2evt(evt_ar2) 
#      
#     evt_td1.show_td(1000,0.01)
#     evt_td2.show_td(1000,0.01) 


# In[]
# 数据预处理: 
# 数据归一化:x,y，t归一化到1
x_max_train = np.max(train_x[:,0,:])
y_max_train = np.max(train_x[:,1,:])
t_max_train = np.max(train_x[:,2,:])

x_max_test = np.max(test_x[:,0,:])
y_max_test = np.max(test_x[:,1,:])
t_max_test = np.max(test_x[:,2,:])

train_x[:,0,:] = train_x[:,0,:]/x_max_train
train_x[:,1,:] = train_x[:,1,:]/y_max_train
train_x[:,2,:] = train_x[:,2,:]/t_max_train

train_y[:,0,:] = train_y[:,0,:]/x_max_train
train_y[:,1,:] = train_y[:,1,:]/y_max_train
train_y[:,2,:] = train_y[:,2,:]/t_max_train

test_x[:,0,:] = test_x[:,0,:]/x_max_test
test_x[:,1,:] = test_x[:,1,:]/y_max_test
test_x[:,2,:] = test_x[:,2,:]/t_max_test

test_y[:,0,:] = test_y[:,0,:]/x_max_test
test_y[:,1,:] = test_y[:,1,:]/y_max_test
test_y[:,2,:] = test_y[:,2,:]/t_max_test


train_x[:,3,:] = train_x[:,3,:]+1 #将-1,0,1的极性表示转换为0,1,2以适应cross_entropyloss
X_train = train_x.transpose(0,2,1) # X变为 样本数 * n_step * 特征维度

train_y[:,3,:] = train_y[:,3,:]+1
y_train =  train_y[:,3,:]

test_x[:,3,:] = test_x[:,3,:]+1
X_test = test_x.transpose(0,2,1) 

test_y[:,3,:] = test_y[:,3,:]+1
y_test = test_y[:,3,:]


#修改步长
X_train = X_train.reshape((-1,n_steps,n_inputs))
y_train = y_train.reshape((-1,n_steps))
X_test = X_test.reshape((-1,n_steps,n_inputs))
y_test = y_test.reshape((-1,n_steps))

print('X_train: ', X_train.shape, '\ty_train: ', y_train.shape)
print('X_test: ',X_test.shape, '\ty_test: ', y_test.shape)

#将X_test变成【样本数 步长 维数】； y_test展平成 样本数*步长 的一维数组
y_test1 = y_test.flatten()   #测试时输入标签


# 对时间维度进行归一化
if time_std==0:
    X_test[:,:,2] = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in X_test[:,:,2]])  #对时间维度进行“步归一化”
if time_std==1:
    X_test[:,:,2] = np.array((X_test[:,:,2] - np.min(X_test[:,:,2]))/(np.max(X_test[:,:,2]) -np.min(X_test[:,:,2]))) #时间维度“批归一化”



# In[] 模型加载
# 初始化
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())    

# tensorboard
writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# 加载
#layers = [op.name for op in graph.get_operations() if op.type == 'train' in op.name]   找op的方法  
restorer = tf.train.import_meta_graph(old_model_path+old_model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(old_model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_index])

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("inputs/Placeholder:0")
y = graph.get_tensor_by_name("inputs/Placeholder_1:0")
merged = graph.get_tensor_by_name("Merge/MergeSummary:0")       
outputs = graph.get_tensor_by_name("outputs/Reshape:0")
accuracy = graph.get_tensor_by_name("evaluation/Mean_1:0")  
training_op = graph.get_operation_by_name('train/Adam')

acc = sess.run(accuracy, feed_dict={X: X_test, y:y_test1})  #for SeqRNN2--LSTM
print('\nfinetune mode:\ncurrent test accuracy: ', acc, '\n\nfinetuning start!\n\n')

# saver初始化
saver = tf.train.Saver()

# In[]
# 训练
for epoch in range(n_epochs):
    for x_batch, y_batch in my_io.batch_iter(batch_size, X_train, y_train):      
        x_batch=x_batch.reshape((-1,n_steps,n_inputs))#转换成batch_size个n_steps*n_inputs的输入
        # 时间归一化
        if time_std==0:
            x_batch[:,:,2] = np.array([(xi - np.min(xi))/(np.max(xi) -np.min(xi)) for xi in x_batch[:,:,2]])  #对时间维度进行“步归一化”
        if time_std==1:
            x_batch[:,:,2] = np.array((x_batch[:,:,2] - np.min(x_batch[:,:,2]))/(np.max(x_batch[:,:,2]) -np.min(x_batch[:,:,2]))) #时间维度“批归一化”        
        y_batch = y_batch.flatten() #zzh:确定一下展开顺序对不对
        sess.run(training_op,feed_dict={X:x_batch,y:y_batch})      

    if epoch%3 == 0:
        tr, acc_train = sess.run([merged, accuracy], feed_dict={X:x_batch,y:y_batch})
        # te, acc_test = sess.run([merged, accuracy], feed_dict={X:X_test[0:10000,...],y:y_test1[0:640000,...]})     
        te, acc_test = sess.run([merged, accuracy], feed_dict={X:X_test, y:y_test1})         
        writer_tr.add_summary(tr, epoch)
        writer_te.add_summary(te, epoch)  
        t2 = time()
        print(epoch,"Train accuracy:", acc_train, "Test accuracy:", acc_test, "Time cost:%.2f min"%((t2-t1)/60))    
    
    if epoch%20 == 0 and epoch != 0:
        saver.save(sess, model_path+model_name,global_step=epoch, write_meta_graph=False)
        print('epoch %d model saved to:'%epoch, model_path+'my_model')
saver.save(sess,model_path+model_name) 
print('epoch: %d model saved to:'%epoch, model_path+model_name) 


# In[ ]:
## 预测
is_Predict = False

if is_Predict:
    x_data = np.copy(X_test)
    
    ydata = np.copy(y_test)
    
    x_data_ =  np.copy(X_test)
    ypreds_ = sess.run(outputs, feed_dict={X: x_data_})
    
    
    #preds = np.squeeze(preds_)
    preds_data =  np.copy(x_data)  #注意复制时要deepcopy才行
    preds_data[:,:,3] = ypreds_
    
    gt_data =  np.copy(x_data)
    gt_data[:,:,3] = ydata
    
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
    
    
    for i in range(20):
        i = int(i*len(X_test)/20 + 50)
        x_data_i = x_data[i,...]     
        preds_data_i = preds_data[i,...]
        gt_data_i = gt_data[i,...]
        
        evt_data = mc.array2evt(x_data_i, 160, 160)    
        evt_preds = mc.array2evt(preds_data_i, 160, 160)
        evt_data_gt = mc.array2evt(gt_data_i, 160, 160)
        
        evt_data.show_td(400,0.01)
        sleep(0.5)    
        evt_preds.show_td(400,0.01)
        sleep(0.5)    
        evt_data_gt.show_td(400,0.01)    
        sleep(1)
    

# In[]

# 终止化

#sess.close()
#writer_tr.close()
#writer_te.close()
        
       
# In[]

# 测试代码
#outputs = tf.reshape(hidden_outputs,[-1, n_neurons])
#logits = fully_connected(outputs,n_outputs,activation_fn=None)
#[50:10000:1000]
#y_batch = y_batch.reshape((-1, ))           
    