# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:01:43 2019

双层单向GRU + attention mechanism

[0] SeqRNN_6
[0->1 SeqRNN_6_attn]
+1: attention mechanism
-+2: dropout
-+3: step standardlization
@author: zzh server code
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
import time

# In[] 
# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

# In[]
#重置计算图
tf.reset_default_graph()  

# In[]
#参数设置
n_steps=64   #步长
n_inputs=4 #输入数据个数(特征维度)
# encoder_n_neurons = 32  #每层神经元的数量
encoder_n_neurons = [32, 32] #encoder 每层神经元的数量
decoder_n_neurons = [32, 32] #decoder 每层神经元的数量
attention_depth = decoder_n_neurons[-1]
n_outputs=3  #输出数据（三种输出分别代�?1,0,1, 此处实际输出为类别序号，分别�?,1,2�?
n_epochs=200
batch_size=128
learning_rate=0.001

# 标志设置
t1 = time.time() #计时起点

# 路径设置
path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/full scale/N_MNIST_seq_train.mat"
path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_seq/full scale/N_MNIST_seq_test.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "/data/zzh/model_data/SeqRNN/"
model_dir = "SeqRNN_6_attn--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/SeqRNN_6_attn_'+timestamp
test_log_dir = 'logs/test/SeqRNN_6_attn_'+timestamp

# In[ ]:
# 加载数据
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

train_x = train_data['N_MNIST_seq_train'].astype('float32')
train_y = train_data['N_MNIST_seq_train_gt'].astype('float32')
test_x = test_data['N_MNIST_seq_test'].astype('float32')
test_y = test_data['N_MNIST_seq_test_gt'].astype('float32')

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
# 数据预处�? 
# 数据归一�?x,y，t归一化到1
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


train_x[:,3,:] = train_x[:,3,:]+1 #�?1,0,1的极性表示转换为0,1,2以适应cross_entropyloss
X_train = train_x.transpose(0,2,1) # X变为 样本�?* n_step * 特征维度

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

#将X_test变成【样本数 步长 维数】； y_test展平�?样本�?步长 的一维数�?
y_test1 = y_test.flatten()   #训练时验证输�?

# In[]

# 网络构建
with tf.name_scope('inputs'):
    X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    y=tf.placeholder(tf.int32,[None])

#he_init=tf.contrib.layers.variance_scaling_initializer()#He initialization 参数初始化
#with tf.variable_scope("rnn",initializer=he_init):】
with tf.name_scope('encoder'):   
    # 1-RNN
#    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=encoder_n_neurons,activation = tf.nn.relu) #,activation = tf.nn.relu/tanh   
    
    # 1-LSTM
    # basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=encoder_n_neurons, forget_bias=0.8, activation = tf.nn.relu) #,activation = tf.nn.relu/tanh       
    # hiddens0, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32) #zzh    
    
    # 2-RNN
    cells = [tf.contrib.rnn.GRUCell(num_units=n) for n in encoder_n_neurons]   #activation = tf.nn.relu/tanh
    stacked_rnn_cell =  tf.contrib.rnn.MultiRNNCell(cells)
    hiddens0, states = tf.nn.dynamic_rnn(stacked_rnn_cell, X, dtype=tf.float32) #zzh 

with tf.name_scope('decoder'):  
    decoder_layers = [tf.contrib.rnn.GRUCell(num_units=n) for n in decoder_n_neurons]
    decoder = tf.contrib.rnn.MultiRNNCell(decoder_layers)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=attention_depth,
        memory=hiddens0)

    attn_decoder = tf.contrib.seq2seq.AttentionWrapper(
        decoder, attention_mechanism,
        # cell_input_fn=lambda inputs, attention: inputs,  # disable input-feeding mechanism
        alignment_history=True, output_attention=True)

    fc_layer = tf.layers.Dense(n_inputs)
    training_helper = tf.contrib.seq2seq.TrainingHelper(y_train, [n_steps])

    decoder_initial_state = attn_decoder.zero_state(batch_size, tf.float32).clone(
        cell_state=states)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=attn_decoder, helper=training_helper,
        initial_state=decoder_initial_state, output_layer=fc_layer)

    logits, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(training_decoder)

    # decoder_logits: [B, T, V]
    decoder_logits = logits.rnn_output
    # [decoder_steps, batch_size, encoder_steps]
    attention_matrices = final_state.alignment_history.stack(
        name="train_attention_matrix")
    print("logits: ", decoder_logits)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=decoder_logits)

# mask = tf.sequence_mask(decoder_length,
#                         maxlen=tf.reduce_max(decoder_length),
#                         dtype=tf.float32)

# loss = tf.reduce_sum(stepwise_cross_entropy * mask) / tf.reduce_sum(mask)
# tf.summary.scalar("loss", loss)
# train_op = tf.train.AdamOptimizer().minimize(loss)
# merged_summary = tf.summary.merge_all()



with tf.name_scope('outputs'):
    outputs = tf.reshape(tf.argmax(decoder_logits,1),[-1, n_steps])
    
with tf.name_scope('evaluation'):
    loss=tf.reduce_mean(xentropy)
    tf.summary.scalar('loss', loss)
    
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))     
    tf.summary.scalar('accuracy', accuracy)
    
with tf.name_scope('train'):        
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)



# In[]
# 初始化会话和模型保存器及tensorboard
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# 训练
for epoch in range(n_epochs):
    for x_batch, y_batch in my_io.batch_iter(batch_size, X_train, y_train):      
        x_batch=x_batch.reshape((-1,n_steps,n_inputs))#转换成batch_size个n_steps*n_inputs的输�?
        y_batch = y_batch.flatten() #zzh:确定一下展开顺序对不�?
        sess.run(training_op,feed_dict={X:x_batch,y:y_batch}) 
       
    if epoch%3 == 0:
        tr, acc_train = sess.run([merged, accuracy], feed_dict={X:x_batch,y:y_batch})
        # te, acc_test = sess.run([merged, accuracy], feed_dict={X:X_test[0:10000,...],y:y_test1[0:640000,...]})     
        te, acc_test = sess.run([merged, accuracy], feed_dict={X:X_test, y:y_test1})         
        writer_tr.add_summary(tr, epoch)
        writer_te.add_summary(te, epoch)  
        t2 = time.time()
        print(epoch,"Train accuracy:", acc_train, "Test accuracy:", acc_test, "Time cost:%.2f min"%((t2-t1)/60))    
    
    if epoch%20 == 0 and epoch != 0:
        saver.save(sess, model_path+'my_model',global_step=epoch, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%epoch, model_path+'my_model')
saver.save(sess,model_path+'my_model') 
print('epoch: %d model saved to:'%epoch, model_path+'my_model')             
        

# In[ ]:
# 预测

# x_data = np.copy(X_test)
# ydata = np.copy(y_test)

# x_data_ =  np.copy(X_test)
# ypreds_ = sess.run(outputs, feed_dict={X: x_data_})

# preds_data =  np.copy(x_data)  #注意复制时要deepcopy才行
# preds_data[:,:,3] = ypreds_

# gt_data =  np.copy(x_data)
# gt_data[:,:,3] = ydata

# # 转换为【样本数*特征维数*样本长度】的形式
# x_data = x_data.transpose(0,2,1)
# preds_data =preds_data.transpose(0,2,1)
# gt_data = gt_data.transpose(0,2,1)

# # 去归一�?
# x_data[:,0,:] = x_data[:,0,:]*x_max_test
# x_data[:,1,:] = x_data[:,1,:]*y_max_test
# x_data[:,2,:] = x_data[:,2,:]*t_max_test

# preds_data[:,0,:] = preds_data[:,0,:]*x_max_test
# preds_data[:,1,:] = preds_data[:,1,:]*y_max_test
# preds_data[:,2,:] = preds_data[:,2,:]*t_max_test

# gt_data[:,0,:] = gt_data[:,0,:]*x_max_test
# gt_data[:,1,:] = gt_data[:,1,:]*y_max_test
# gt_data[:,2,:] = gt_data[:,2,:]*t_max_test

 
# preds_data[:,3,:] = preds_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式
# x_data[:,3,:] = x_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式
# gt_data[:,3,:] = gt_data[:,3,:] - 1; #将极性表示还原为-1,0,1形式


# for i in range(20):
#    i = int(i*len(X_test)/20 + 90)
#    x_data_i = x_data[i,...]     
#    preds_data_i = preds_data[i,...]
#    gt_data_i = gt_data[i,...]
   
#    evt_data = mc.array2evt(x_data_i, 160, 160)    
#    evt_preds = mc.array2evt(preds_data_i, 160, 160)
#    evt_data_gt = mc.array2evt(gt_data_i, 160, 160)
   
#    evt_data.show_td(400,0.01)
#    sleep(0.5)    
#    evt_preds.show_td(400,0.01)
#    sleep(0.5)    
#    evt_data_gt.show_td(400,0.01)    
#    sleep(1)
    
      


# In[]
# release
#sess.close()
#writer_tr.close()
#writer_te.close()
        
       
# In[]
# 测试代码
#outputs = tf.reshape(hidden_outputs,[-1, encoder_n_neurons])
#logits = fully_connected(outputs,n_outputs,activation_fn=None)
#[50:10000:1000]
#y_batch = y_batch.reshape((-1, ))           
    