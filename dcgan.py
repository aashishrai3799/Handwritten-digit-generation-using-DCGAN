# -*- coding: utf-8 -*-

# Digit Generation using DCGAN

####install scipy(v1.2)
"""

#!pip install scipy==1.2

"""###Import Libraries


*   tensorflow__1.14
*   numpy______1.16.4
*   matplotlib___3.0.3
*   scipy________1.2.0
*   os
*   time
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import scipy.misc
import scipy
import time
from tensorflow.examples.tutorials.mnist import input_data

"""###Download and extract MNIST dataset"""

mnist = input_data.read_data_sets("MNIST/data", one_hot=False)

"""### Generator

*   input: random vector
*   output: 32x32 image
"""

def generator(z):
    
    
    gen = tf.contrib.layers.fully_connected(z,4*4*256,normalizer_fn=tf.contrib.layers.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    gen = tf.reshape(gen,[-1,4,4,256])
    
    
    gen1 = tf.layers.conv2d_transpose(gen, 64, 5, 2, padding = 'SAME')
    gen1 = tf.nn.relu(gen1)
    
    gen2 = tf.layers.conv2d_transpose(gen1, 32, 5, 2, padding = 'SAME')
    gen2 = tf.nn.relu(gen2)
    
    gen3 = tf.layers.conv2d_transpose(gen2, 16, 5, 2, padding = 'SAME')
    gen3 = tf.nn.relu(gen3)
    
    g_out = tf.contrib.layers.convolution2d_transpose(\
        gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
        biases_initializer=None,activation_fn=tf.nn.tanh,\
        scope='g_out', weights_initializer=initializer)
    
    return g_out

"""### Discriminator

*   input: 32x32 image
*   output: probability of being real or fake image
"""

def discriminator(x, reuse=False):
    

    dis1 = tf.contrib.layers.convolution2d(x, 16, [4,4], stride=[2,2], padding="SAME",\
        biases_initializer=None, activation_fn = tf.nn.leaky_relu,\
        reuse=reuse, scope='d_conv1', weights_initializer=initializer)
    
    dis2 = tf.contrib.layers.convolution2d(dis1,32,[4,4], stride=[2,2], padding="SAME",\
        normalizer_fn=tf.contrib.layers.batch_norm,activation_fn=tf.nn.leaky_relu,\
        reuse=reuse, scope='d_conv2', weights_initializer=initializer)

    dis3 = tf.contrib.layers.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=tf.contrib.layers.batch_norm,activation_fn=tf.nn.leaky_relu,\
        reuse=reuse,scope='d_conv3',weights_initializer=initializer)
    
    d_out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(dis3),1,activation_fn=tf.nn.sigmoid,\
        reuse=reuse,scope='d_out', weights_initializer=initializer)
    
    return d_out

"""### UPDATING VARIABLES"""

tf.reset_default_graph()
z_vector = 100 #Size of z vector used for generator.
initializer = tf.truncated_normal_initializer(stddev = 0.02)

z_in = tf.placeholder(shape = [None, z_vector], dtype = tf.float32) #Random vector
real_in = tf.placeholder(shape = [None, 32, 32, 1], dtype = tf.float32) #Real images

Gz = generator(z_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gz, reuse = True) #Produces probabilities for generator images

d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

trainerD = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.00005)
trainerG = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.00005)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #update the weights for the generator network.

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

"""###Save Image Function"""

def save_images(images, size, path):
  images = (images+1.)/2.
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1]))

  for ind, image in enumerate(images):
    ii = ind % size[1]
    jj = ind // size[1]
    img[jj * h : jj * h + h, ii * w : ii * w + w] = image

  if i % 1000 == 0:    # DISPLAY IMAGE IF EPOCH IS MULTIPLE OG 1000
    plt.imshow(img)
    plt.show() 
  return scipy.misc.imsave(path, img)

"""### TRAINING NETWORK"""

batch_size = 256                   # Size of image batch to apply at each iteration
epochs = 160000                      # Total number of iterations to use
sample_directory = './MNIST/figs'   # Directory to save generated sample images
model_directory = './MNIST/models'  # Directory to save trained model to

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    
    t1 = 0      # initializing TIME VARIABLE 1
    t2 = 0      # initializing TIME VARIABLE 2
    
    for i in range(epochs):
        zs = np.random.uniform(-1.0, 1.0, size = [batch_size, z_vector]).astype(np.float32) #Generate a random z batch
        xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
        xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0), (2,2), (2,2), (0,0)), 'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) #Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
        
        if i % 100 == 0:
            t2 = time.time()
            print (i, ':  G_Loss: ' + str(gLoss)[:6] + '  D_Loss: ' + str(dLoss)[:6], ' Time:', round((t2-t1), 4), 'sec')
            if '0.0' in str(dLoss)[:6] :       # BREAK LOOP IF DLOSS APPROACHES 0.0
              break
              
            z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_vector]).astype(np.float32) #Generate another z batch
            newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
            
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            
#  S A V I N G  I M A G E            
            if i % 10000 == 0:
              save_images(np.reshape(newZ[0:36],[36,32,32]),[6,6],sample_directory + '/fig' + str(i) + '.png')
            t1 = time.time()

        
#  S A V I N G  M O D E L        

        if i % 5000 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory + '/model-' + str(i) + '.cptk')
            print ("Model Saved:", str(i) + '.cptk')

"""## Generate Sample Image from Generated Model"""

path = '/content/MNIST/models'
sample_directory = './MNIST/sample_fig' #Directory to save sample images from generator in.
model_directory = './MNIST/models' #Directory to load trained model from.
batch_size_sample = 36
i = 190000
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    print ('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
    zs = np.random.uniform(-1.0, 1.0, size = [batch_size_sample, z_vector]).astype(np.float32) #Generate a random z batch
    newZ = sess.run(Gz, feed_dict = {z_in:z2}) #Use new z to get sample images from generator.
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    save_images(np.reshape(newZ[0:batch_size_sample], [36, 32, 32]), [6, 6], sample_directory+'/fig'+str(i)+'.png')

