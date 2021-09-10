'''
【累积重构-卷积去噪自编码】

数据集：N_MNIST_pic

问题：
1. 模型训练可能出现过拟合(outputs极性化后求cost； dropout，见ConvDAE_3_1)
2. 显示时极性化阈值的选取很重要
'''

# In[45]:
#导入基本模块
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import os

# In[8]:


#导入自定义模块
from my_tf_lib import my_io


# In[9]:


#重置tensorboard graph
tf.reset_default_graph() 

# In[9]:
# 路径
timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "ConvDAE_3--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/ConvDAE_3_'+timestamp
test_log_dir = 'logs/test/ConvDAE_3_'+timestamp
# In[57]:


path1 = r"D:\1-Codes\matlab\resource\dataset\N_MNIST_pic\N_MNIST_pic_train.mat"
path2 = r"D:\1-Codes\matlab\resource\dataset\N_MNIST_pic\N_MNIST_pic_test.mat"
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)
train_x = train_data['N_MNIST_pic_train'].astype('float32')
train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
test_x = test_data['N_MNIST_pic_test'].astype('float32')
test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)


# In[11]:


# 数据打印测试

for k in range(5):
    plt.subplot(2,5,k+1)
    plt.imshow(train_x[k])
    plt.title('train_x_%d'%(k+1))
    plt.xticks([])
    plt.yticks([])        
    plt.subplot(2,5,k+6)
    plt.imshow(train_y[k])
    plt.title('train_y_%d'%(k+1))
    plt.xticks([])
    plt.yticks([])


# # 构造模型

# ## 输入

# In[13]:


with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')


# ## Encoder
# 
# 三层卷积

# In[14]:


#选择激活函数
# act_fun = tf.nn.relu 
act_fun = tf.nn.tanh
act_fun_out = tf.nn.tanh


# In[15]:


with tf.name_scope('encoder'):
    conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=act_fun)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=act_fun)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=act_fun)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')


# ## Decoder

# In[16]:


with tf.name_scope('decoder'):
    conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=act_fun)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=act_fun)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=act_fun)


# ## logits and outputs

# In[17]:


with tf.name_scope('outputs'):
    logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

    # outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
    outputs_ = act_fun_out(logits_, name='outputs_')


# ## loss and Optimizer

# In[18]:


with tf.name_scope('loss'):
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
#     loss = tf.reduce_sum(tf.square(targets_ -  outputs_))

    loss = tf.losses.mean_squared_error(targets_ , outputs_)


    cost = tf.reduce_mean(loss)
    tf.summary.scalar('cost', cost)
    test_cost = tf.reduce_mean(loss)
    tf.summary.scalar('test_cost', test_cost)
    
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# # 训练

# In[19]:


sess = tf.Session()


# In[20]:


# parameters

epochs = 10
batch_size = 128
sess.run(tf.global_variables_initializer())


# In[21]:


# tensorboard

merged = tf.summary.merge_all() 

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
train_log_dir = 'logs/train/'+timestamp
test_log_dir = 'logs/test/'+timestamp

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_ts = tf.summary.FileWriter(test_log_dir)


# In[22]:


# trainning


for e in range(epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y):
#         x = abs(batch_x.reshape((-1, 28, 28, 1)))  #将负值像素取绝对值
#         y = abs(batch_y.reshape((-1, 28, 28, 1)))
        x = batch_x.reshape((-1, 28, 28, 1))
        y = batch_y.reshape((-1, 28, 28, 1))

        batch_cost, _ = sess.run([cost, optimizer],
                           feed_dict={inputs_: x,
                                     targets_: y})
        print("Epoch: {}/{} ".format(e+1, epochs),
             "Training loss: {:.4f}".format(batch_cost))
    
    test_x1 = test_x[500:10000:1000].reshape((-1, 28, 28, 1))
    test_y1 = test_y[500:10000:1000].reshape((-1, 28, 28, 1))    
    test_cost_v = sess.run(test_cost, feed_dict={inputs_: test_x1, targets_: test_y1})
    print("Epoch: {}".format(e+1),
         "Test loss: {:.4f}".format(test_cost_v))    
    
    tr = sess.run(merged, feed_dict={inputs_: x, targets_: y})
    writer_tr.add_summary(tr, e)
    
    ts = sess.run(merged, feed_dict={inputs_: test_x1, targets_: test_y1})
    writer_ts.add_summary(ts, e)    


# In[61]:


fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))

# k=10
# in_imgs = abs(test_x[k:k+10])
# gt_imgs = abs(test_y[k:k+10])
# in_imgs = test_x[k:k+10]
# gt_imgs = test_y[k:k+10]

# ConvDAE3
in_imgs = test_x[600:10000:1000]
gt_imgs = test_y[600:10000:1000]

reconstructed = sess.run(outputs_, 
                         feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})

# zzh:极性化，阈值-0.5，0.5        
thresh = 0.3
polarization = 0
if polarization:     
    reconstructed[reconstructed<=-1*thresh] = -1.
    reconstructed[reconstructed>=thresh] = 1.
    reconstructed[(reconstructed>-1*thresh) & (reconstructed<thresh)] = 0.



for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[ ]:
saver = tf.train.Saver()
saver.save(sess,model_path+'my_model') 
print('model saved to', model_path+'my_model')
# In[24]:
# release
#sess.close()





