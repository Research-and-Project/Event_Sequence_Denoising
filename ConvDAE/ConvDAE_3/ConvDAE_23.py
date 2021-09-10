'''
【累积重构-卷积去噪自编码】

 数据集：训练N_MNIST——diff， 测试N_MNIST_pic
 
 问题：
1. 考虑加入outputs极性化后求cost；
'''


# In[59]:


#导入基本模块
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from my_tf_lib import my_io
import os


# In[62]:
#重置tensorboard graph
tf.reset_default_graph() 


# In[62]:
# 路径
timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "ConvDAE_23--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/ConvDAE_23_'+timestamp
test_log_dir = 'logs/test/ConvDAE_23_'+timestamp

path1 = r'D:\1-Codes\matlab\resource\dataset\MNIST_diff\MNIST_diff_train.mat'
path2 = r'D:\1-Codes\matlab\resource\dataset\MNIST_diff\MNIST_diff_test.mat'
path3 = r"D:\1-Codes\matlab\resource\dataset\N_MNIST_pic\N_MNIST_pic_test.mat"

# 参数
epochs = 10
batch_size = 128


# In[63]:
# 加载数据
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)
real_data = my_io.load_mat(path3)

train_x = train_data['train_diff'].astype('float32')
train_y = train_data['train_diff_gt'].astype('float32')
test_x = test_data['test_diff'].astype('float32')
test_y = test_data['test_diff_gt'].astype('float32')
real_x = real_data['N_MNIST_pic_test'].astype('float32')
real_y = real_data['N_MNIST_pic_test_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape,
     '\nreal_x: ', real_x.shape, '\ttest_y: ', real_y.shape)


# In[64]:
# 数据打印测试
#for k in range(5):
#    plt.subplot(2,5,k+1)
#    plt.imshow(train_x[k])
#    plt.title('train_x_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])        
#    plt.subplot(2,5,k+6)
#    plt.imshow(train_y[k])
#    plt.title('train_y_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])




# In[65]:
# 构造模型

# 输入
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')


#选择激活函数
# act_fun = tf.nn.relu 
act_fun = tf.nn.tanh
act_fun_out = tf.nn.tanh


# Encoder
with tf.name_scope('encoder'):
    conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=act_fun)
    conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=act_fun)
    conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=act_fun)
    conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')


# Decoder
with tf.name_scope('decoder'):
    conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
    conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=act_fun)

    conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=act_fun)

    conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=act_fun)


# logits and outputs
with tf.name_scope('outputs'):
    logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

    # outputs_ = tf.nn.sigmoid(logits_, name='outputs_')
    outputs_ = act_fun_out(logits_, name='outputs_')


# loss and Optimizer
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



# In[72]:
# 初始化
sess = tf.Session()    
sess.run(tf.global_variables_initializer())

# tensorboard
merged = tf.summary.merge_all() 
writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_ts = tf.summary.FileWriter(test_log_dir)


# In[74]:
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
    
    test_xx = test_x[500:10000:100].reshape((-1, 28, 28, 1))
    test_yy = test_y[500:10000:100].reshape((-1, 28, 28, 1))    
    test_cost_v = sess.run(test_cost, feed_dict={inputs_: test_xx, targets_: test_yy})
    print("Epoch: {}".format(e+1),
         "Test loss: {:.4f}".format(test_cost_v))    
    
    tr = sess.run(merged, feed_dict={inputs_: x, targets_: y})
    writer_tr.add_summary(tr, e)
    
    ts = sess.run(merged, feed_dict={inputs_: test_xx, targets_: test_yy})
    writer_ts.add_summary(ts, e)    


# In[77]:
# 测试

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))

in_imgs = real_x[100:10000:1000]
gt_imgs = real_y[100:10000:1000]

feed_dict = {inputs_:in_imgs.reshape((-1, 28, 28, 1)), targets_: gt_imgs.reshape((-1, 28, 28, 1))}
test_cost_v, reconstructed = sess.run([test_cost,outputs_], feed_dict=feed_dict)
print('current cost:', test_cost_v)
# zzh:极性化，阈值-0.5，0.5        
thresh = 0.35
polarization = 1
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
# 保存模型
saver = tf.train.Saver()
saver.save(sess,model_path+'my_model') 
print('model saved to', model_path+'my_model')


# In[76]:
# release
#sess.close()




