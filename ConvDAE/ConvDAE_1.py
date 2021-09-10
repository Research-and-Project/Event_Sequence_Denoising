
# coding: utf-8

# # 卷积自编码器
# 
# 在之前，我们实现了一个简单的自编码器来对图像进行复现。现在我们通过增加卷积层来提高我们自编码器的复现能力。

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


print("TensorFlow Version: %s" % tf.__version__)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # 加载数据

# In[33]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)


# In[ ]:


img = mnist.train.images[10]
plt.imshow(img.reshape((28, 28)))


# # 构造模型

# ## 输入

# In[6]:


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')


# ## Encoder
# 
# 三层卷积

# In[7]:


conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')


# ## Decoder

# In[8]:


conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)


# ## logits and outputs

# In[9]:


logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

outputs_ = tf.nn.sigmoid(logits_, name='outputs_')


# ## loss and Optimizer

# In[10]:


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# # 训练

# In[11]:


sess = tf.Session()


# In[36]:


noise_factor = 0.7
epochs = 5
batch_size = 128
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1)) #原数据是拉平的，reshape为二维图像

        
        # 加入噪声
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        #zzh:打乱加噪图片，使得加噪图片和ground truth不对应，观察影响
        #tf.random_shuffle(noisy_imgs)
        
        # zzh:二值化，阈值0.5        
        imgs[imgs<0.5] = 0.
        imgs[imgs>=0.5] = 1.               
        noisy_imgs[noisy_imgs<0.5] = 0.
        noisy_imgs[noisy_imgs>=0.5] = 1.
        
        batch_cost, _ = sess.run([cost, optimizer],
                           feed_dict={inputs_: noisy_imgs,
                                     targets_: imgs})
        
        print("Epoch: {}/{} ".format(e+1, epochs),
             "Training loss: {:.4f}".format(batch_cost))


# In[37]:


fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[10:20]
   
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
#noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)*1.3 #zzh:改变噪声分布
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
      
# zzh:二值化，阈值0.5        
in_imgs[in_imgs<0.5] = 0.
in_imgs[in_imgs>=0.5] = 1.               
noisy_imgs[noisy_imgs<0.5] = 0.
noisy_imgs[noisy_imgs>=0.5] = 1.
      
reconstructed = sess.run(outputs_, 
                         feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

# zzh:二值化，阈值0.5        
reconstructed[reconstructed<0.5] = 0.
reconstructed[reconstructed>=0.5] = 1.

for images, row in zip([noisy_imgs, reconstructed, in_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)


# In[14]:


sess.close()

