# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:34:13 2018

@author: Aditya
"""


# coding: utf-8

# # Project: **German Traffic Sign Classification Using TensorFlow** 
# **In this project, I used Python and TensorFlow to classify traffic signs.**
# 
# **Dataset used: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# This dataset has more than 50,000 images of 43 classes.**
# 
# **I was able to reach a +99% validation accuracy, and a 97.3% testing accuracy.**

# ## Pipeline architecture:
# - **Load The Data.**
# - **Dataset Summary & Exploration**
# - **Data Preprocessing**.
#     - Shuffling.
#     - Grayscaling.
#     - Local Histogram Equalization.
#     - Normalization.
# - **Design a Model Architecture.**
#     - LeNet-5.
#     - VGGNet.
# - **Model Training and Evaluation.**
# - **Testing the Model Using the Test Set.**
# - **Testing the Model on New Images.**
# 
# I'll explain each step in details below.

# #### Environement:
# -  Ubuntu 16.04
# -  Anaconda 5.0.1
# -  Python 3.6.2
# -  TensorFlow 0.12.1 (GPU support)

# In[1]:


# Importing Python libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
import math
from scipy.interpolate import splprep, splev


# ---
# ## Step 1: Load The Data

# Download the dataset from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
# 
# We already have three `.p` files of 32x32 resized images:
# - `train.p`: The training set.
# - `test.p`: The testing set.
# - `valid.p`: The validation set.
# 
# We will use Python `pickle` to load the data.

# In[2]:


training_file = "./traffic-signs-data/train.p"
validation_file= "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


# In[3]:


# Mapping ClassID to traffic sign names
signs = []
with open('signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()


# ---
# 
# ## Step 2: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.
# 
# The code snippets below will provide a basic summery of the Dataset.

# **First, we will use `numpy` provide the number of images in each subset, in addition to the image size, and the number of unique classes.**

# In[4]:


X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples
n_train = X_train.shape[0]

# Number of testing examples
n_test = X_test.shape[0]

# Number of validation examples.
n_validation = X_valid.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples: ", n_train)
print("Number of testing examples: ", n_test)
print("Number of validation examples: ", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# **Then, we will use `matplotlib` plot sample images from each subset.**

# In[5]:


def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: An np.array compatible with plt.imshow.
            lanel (Default = No label): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        indx = random.randint(0, len(dataset))
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap = cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    #plt.show()


# In[7]:


# Plotting sample examples
list_images(X_train, y_train, "Training example")
list_images(X_test, y_test, "Testing example")
list_images(X_valid, y_valid, "Validation example")


# **And finally, we will use `numpy` to plot a histogram of the count of images in each unique class.**

# In[8]:


def histogram_plot(dataset, label):
    """
    Plots a histogram of the input data.
        Parameters:
            dataset: Input data to be plotted as a histogram.
            lanel: A string to be used as a label for the histogram.
    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
   # plt.show()


# In[9]:


# Plotting histograms of the count of each sign
histogram_plot(y_train, "Training examples")
histogram_plot(y_test, "Testing examples")
histogram_plot(y_valid, "Validation examples")


# ---
# 
# ## Step 3: Data Preprocessing
# 
# In this step, we will apply several preprocessing steps to the input images to achieve the best possible results.
# 
# **We will use the following preprocessing techniques:**
# 1. Shuffling.
# 2. Grayscaling.
# 3. Local Histogram Equalization.
# 4. Normalization.

# 1.
# **Shuffling**: In general, we shuffle the training data to increase randomness and variety in training dataset, in order for the model to be more stable. We will use `sklearn` to shuffle our data.

# In[10]:


X_train, y_train = shuffle(X_train, y_train)


# 2.
# **Grayscaling**: In their paper ["Traffic Sign Recognition with Multi-Scale Convolutional Networks"](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) published in 2011, P. Sermanet and Y. LeCun stated that using grayscale images instead of color improves the ConvNet's accuracy. We will use `OpenCV` to convert the training images into grey scale.

# In[11]:


def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# In[12]:


# Sample images after greyscaling
gray_images = list(map(gray_scale, X_train))
list_images(gray_images, y_train, "Gray Scale image", "gray")


# 3.
# **Local Histogram Equalization**: This technique simply spreads out the most frequent intensity values in an image, resulting in enhancing images with low contrast. Applying this technique will be very helpfull in our case since the dataset in hand has real world images, and many of them has low contrast. We will use `skimage` to apply local histogram equalization to the training images.

# In[13]:


def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


# In[14]:


# Sample images after Local Histogram Equalization
equalized_images = list(map(local_histo_equalize, gray_images))
list_images(equalized_images, y_train, "Equalized Image", "gray")


# 4.
# **Normalization**: Normalization is a process that changes the range of pixel intensity values. Usually the image data should be normalized so that the data has mean zero and equal variance.

# In[15]:


def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image


# In[16]:


# Sample images after normalization
n_training = X_train.shape
normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
for i, img in enumerate(equalized_images):
    normalized_images[i] = image_normalize(img)
list_images(normalized_images, y_train, "Normalized Image", "gray")
normalized_images = normalized_images[..., None]


# In[17]:


def preprocess(data):
    """
    Applying the preprocessing steps to the input data.
        Parameters:
            data: An np.array compatible with plt.imshow.
    """
    gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    normalized_images = normalized_images[..., None]
    return normalized_images

#%%
# ---
# 
# ## Step 3: Design a Model Architecture
# 
# In this step, we will design and implement a deep learning model that learns to recognize traffic signs from our dataset [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# We'll use Convolutional Neural Networks to classify the images in this dataset. The reason behind choosing ConvNets is that they are designed to recognize visual patterns directly from pixel images with minimal preprocessing. They automatically learn hierarchies of invariant features at every level from data.
# We will implement two of the most famous ConvNets. Our goal is to reach an accuracy of +95% on the validation set.
# 
# I'll start by explaining each network architecture, then implement it using TensorFlow.

# **Notes**:
# 1. We specify the learning rate of 0.001, which tells the network how quickly to update the weights.
# 2. We minimize the loss function using the Adaptive Moment Estimation (Adam) Algorithm. Adam is an optimization algorithm introduced by D. Kingma and J. Lei Ba in a 2015 paper named [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Adam algorithm computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like [Adadelta](https://arxiv.org/pdf/1212.5701.pdf) and [RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) algorithms, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to [momentum algorithm](http://www.sciencedirect.com/science/article/pii/S0893608098001166?via%3Dihub), which in turn produce better results.
# 3. we will run `minimize()` function on the optimizer which use backprobagation to update the network and minimize our training loss.

# ### 1.  LeNet-5
# LeNet-5 is a convolutional network designed for handwritten and machine-printed character recognition. It was introduced by the famous [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) in his paper [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) in 1998. Although this ConvNet is intended to classify hand-written digits, we're confident it have a very high accuracy when dealing with traffic signs, given that both hand-written digits and traffic signs are given to the computer in the form of pixel images.
# 
# **LeNet-5 architecture:**
# <figure>
#  <img src="LeNet.png" width="1072" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  </figcaption>
# </figure>

# This ConvNet follows these steps:
# 
# Input => Convolution => ReLU => Pooling => Convolution => ReLU => Pooling => FullyConnected => ReLU => FullyConnected
# 
# **Layer 1 (Convolutional):** The output shape should be 28x28x6.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 14x14x6.
# 
# **Layer 2 (Convolutional):** The output shape should be 10x10x16.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 5x5x16.
# 
# **Flattening:** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
# 
# **Layer 3 (Fully Connected):** This should have 120 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 4 (Fully Connected):** This should have 84 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5 (Fully Connected):** This should have 10 outputs.

# In[18]:
'''

class LaNet:  

    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 28x28x6.
        self.filter1_width = 5
        self.filter1_height = 5
        self.input1_channels = 1
        self.conv1_output = 6
        # Weight and bias
        self.conv1_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output),
            mean = self.mu, stddev = self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))
        # Apply Convolution
        self.conv1 = tf.nn.conv2d(x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias
        
        # Activation:
        self.conv1 = tf.nn.relu(self.conv1)
        
        # Pooling: Input = 28x28x6. Output = 14x14x6.
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        # Layer 2 (Convolutional): Output = 10x10x16.
        self.filter2_width = 5
        self.filter2_height = 5
        self.input2_channels = 6
        self.conv2_output = 16
        # Weight and bias
        self.conv2_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),
            mean = self.mu, stddev = self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))
        # Apply Convolution
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_bias
        
        # Activation:
        self.conv2 = tf.nn.relu(self.conv2)
        
        # Pooling: Input = 10x10x16. Output = 5x5x16.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        # Flattening: Input = 5x5x16. Output = 400.
        self.fully_connected0 = flatten(self.conv2)
        
        # Layer 3 (Fully Connected): Input = 400. Output = 120.
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = self.mu, stddev = self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(120))
        self.fully_connected1 = tf.add((tf.matmul(self.fully_connected0, self.connected1_weights)), self.connected1_bias)
        
        # Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)
    
        # Layer 4 (Fully Connected): Input = 120. Output = 84.
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = self.mu, stddev = self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(84))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)
        
        # Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)
    
        # Layer 5 (Fully Connected): Input = 84. Output = 43.
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = self.mu, stddev = self.sigma))
        self.output_bias = tf.Variable(tf.zeros(43))
        self.logits =  tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)

        # Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()
    
    def y_predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset+BATCH_SIZE]
            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, 
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
'''

# ### 2.  VGGNet
# VGGNet was first introduced in 2014 by K. Simonyan and A. Zisserman from the University of Oxford in a paper called [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf). They were investigating the convolutional network depth on its accuracy in the large-scale image recognition setting. Their main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers.
# 
# **VGGNet architecture:**
# <figure>
#  <img src="VGGNet.png" width="1072" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  </figcaption>
# </figure>

# The original VGGNet architecture has 16-19 layers, but I've excluded some of them and implemented a modified version of only 12 layers to save computational resources.
# 
# This ConvNet follows these steps:
# 
# Input => Convolution => ReLU => Convolution => ReLU => Pooling => Convolution => ReLU => Convolution => ReLU => Pooling => Convolution => ReLU => Convolution => ReLU => Pooling => FullyConnected => ReLU => FullyConnected => ReLU => FullyConnected
# 
# **Layer 1 (Convolutional):** The output shape should be 32x32x32.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 2 (Convolutional):** The output shape should be 32x32x32.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 3 (Pooling)** The output shape should be 16x16x32.
# 
# **Layer 4 (Convolutional):** The output shape should be 16x16x64.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5 (Convolutional):** The output shape should be 16x16x64.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 6 (Pooling)** The output shape should be 8x8x64.
# 
# **Layer 7 (Convolutional):** The output shape should be 8x8x128.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 8 (Convolutional):** The output shape should be 8x8x128.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 9 (Pooling)** The output shape should be 4x4x128.
# 
# **Flattening:** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
# 
# **Layer 10 (Fully Connected):** This should have 128 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 11 (Fully Connected):** This should have 128 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 12 (Fully Connected):** This should have 43 outputs.

# In[19]:


class VGGnet:  

    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 32x32x32.
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(32))
        self.conv1   = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b

        # ReLu Activation.
        self.conv1 = tf.nn.relu(self.conv1)

        # Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.
        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(32))
        self.conv2   = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

        # ReLu Activation.
        self.conv2 = tf.nn.relu(self.conv2)

        # Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)

        # Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.
        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = self.mu, stddev = self.sigma))
        self.conv3_b = tf.Variable(tf.zeros(64))
        self.conv3   = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b

        # ReLu Activation.
        self.conv3 = tf.nn.relu(self.conv3)

        # Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.
        self.conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = self.mu, stddev = self.sigma))
        self.conv4_b = tf.Variable(tf.zeros(64))
        self.conv4   = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b

        # ReLu Activation.
        self.conv4 = tf.nn.relu(self.conv4)

        # Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
        self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv) # dropout

        # Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
        self.conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = self.mu, stddev = self.sigma))
        self.conv5_b = tf.Variable(tf.zeros(128))
        self.conv5   = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b

        # ReLu Activation.
        self.conv5 = tf.nn.relu(self.conv5)

        # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
        self.conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = self.mu, stddev = self.sigma))
        self.conv6_b = tf.Variable(tf.zeros(128))
        self.conv6   = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b

        # ReLu Activation.
        self.conv6 = tf.nn.relu(self.conv6)

        # Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
        self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv) # dropout

        # Flatten. Input = 4x4x128. Output = 2048.
        self.fc0   = flatten(self.conv6)

        # Layer 10 (Fully Connected): Input = 2048. Output = 128.
        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(128))
        self.fc1   = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

        # ReLu Activation.
        self.fc1    = tf.nn.relu(self.fc1)
        self.fc1    = tf.nn.dropout(self.fc1, keep_prob) # dropout

        # Layer 11 (Fully Connected): Input = 128. Output = 128.
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = self.mu, stddev = self.sigma))
        self.fc2_b  = tf.Variable(tf.zeros(128))
        self.fc2    = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b

        # ReLu Activation.
        self.fc2    = tf.nn.relu(self.fc2)
        self.fc2    = tf.nn.dropout(self.fc2, keep_prob) # dropout

        # Layer 12 (Fully Connected): Input = 128. Output = n_out.
        self.fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean = self.mu, stddev = self.sigma))
        self.fc3_b  = tf.Variable(tf.zeros(n_out))
        self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b

        # Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()
        
    def y_predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset+BATCH_SIZE]
            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, 
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

# ---
# 
# ## Step 4: Model Training and Evaluation
# 
# In this step, we will train our model using `normalized_images`, then we'll compute softmax cross entropy between `logits` and `labels` to measure the model's error probability.

# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.

# In[20]:


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))


# The `keep_prob` and `keep_prob_conv` variables will be used to control the dropout rate when training the neural network.
# Overfitting is a serious problem in deep nural networks. Dropout is a technique for addressing this problem.
# The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. During training, dropout samples from an exponential number of different “thinned” networks. At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods. This technique was introduced by N. Srivastava, G. Hinton, A. Krizhevsky I. Sutskever, and R. Salakhutdinov in their paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).

# In[22]:


keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers


# In[23]:


# Validation set preprocessing
X_valid_preprocessed = preprocess(X_valid)


# In[24]:


EPOCHS = 30
BATCH_SIZE = 64
DIR = 'Saved_Models'


# Now, we'll run the training data through the training pipeline to train the model.
# - Before each epoch, we'll shuffle the training set.
# - After each epoch, we measure the loss and accuracy of the validation set.
# - And after training, we will save the model.
# - A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# ### LeNet Model

# In[25]:

'''
LeNet_Model = LaNet(n_out = n_classes)
model_name = "LeNet"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training ...")
    print()
    for i in range(EPOCHS):
        normalized_images, y_train = shuffle(normalized_images, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
            sess.run(LeNet_Model.training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})
            
        validation_accuracy = LeNet_Model.evaluate(X_valid_preprocessed, y_valid)
        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
    LeNet_Model.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")

'''
# As we can see, we've been able to reach a maximum accuracy of **95.3%** on the validation set over 30 epochs, using a learning rate of 0.001.
# 
# Now, we'll train the VGGNet model and evaluate it's accuracy.

# ### VGGNet Model

# In[26]:


VGGNet_Model = VGGnet(n_out = n_classes)
model_name = "VGGNet"

# Validation set preprocessing
X_valid_preprocessed = preprocess(X_valid)
one_hot_y_valid = tf.one_hot(y_valid, 43)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        normalized_images, y_train = shuffle(normalized_images, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
            sess.run(VGGNet_Model.training_operation, 
            feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})

        validation_accuracy = VGGNet_Model.evaluate(X_valid_preprocessed, y_valid)
        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
    VGGNet_Model.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")


# Using VGGNet, we've been able to reach a maximum **validation accuracy of 99.3%**. As you can observe, the model has nearly saturated after only 10 epochs, so we can reduce the epochs to 10 and save computational resources.
# 
# We'll use this model to predict the labels of the test set.

# ---
# 
# ## Step 5: Testing the Model using the Test Set
# 
# Now, we'll use the testing set to measure the accuracy of the model over unknown examples.

# In[27]:


# Test set preprocessing
X_test_preprocessed = preprocess(X_test)


# In[42]:


with tf.Session() as sess:
    VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
    y_pred = VGGNet_Model.y_predict(X_test_preprocessed)
    test_accuracy = sum(y_test == y_pred)/len(y_test)
    print("Test Accuracy = {:.1f}%".format(test_accuracy*100))


# ### Test Accuracy = 97.6%

# A remarkable performance!

# Now we'll plot the confusion matrix to see where the model actually fails.

# In[30]:


cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Log of normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.show()


# We observe some clusters in the confusion matrix above. It turns out that the various speed limits are sometimes misclassified among themselves. Similarly, traffic signs with traingular shape are misclassified among themselves. We can further improve on the model using hierarchical CNNs to first identify broader groups (like speed signs) and then have CNNs to classify finer features (such as the actual speed limit).

# ---
# 
# ## Step 6: Testing the Model on New Images
# 
# In this step, we will use the model to predict traffic signs type of 5 random images of German traffic signs from the web our model's performance on these images.

# In[38]:


# Loading and resizing new test images
new_test_images = []
path = './traffic-signs-data/new_test_images/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_test_images.append(img)
new_IDs = [27, 14, 13, 25, 11, 27 ,3 , 17,1,1,1]
print("Number of new testing examples: ", len(new_test_images))


# Displaying the new testing examples, with their respective ground-truth labels:

# In[39]:


plt.figure(figsize=(15, 16))
for i in range(len(new_test_images)):
    plt.subplot(2,len(new_test_images), i+1)
    plt.imshow(new_test_images[i])
    plt.xlabel("New testing image")
    plt.ylabel((i+1))
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
#plt.show()


# These test images include some easy to predict signs, and other signs are considered hard for the model to predict.
# 
# For instance, we have easy to predict signs like the "Stop" and the "No entry". The two signs are clear and belong to classes where the model can predict with  high accuracy.
# 
# On the other hand, we have signs belong to classes where has poor accuracy, like the "Speed limit" sign, because as stated above it turns out that the various speed limits are sometimes misclassified among themselves, and the "Pedestrians" sign, because traffic signs with traingular shape are misclassified among themselves.

# In[40]:


# New test data preprocessing
new_test_images_preprocessed = preprocess(np.asarray(new_test_images))


# In[41]:

DIR = "Saved_Models"
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers
VGGNet_Model = VGGnet(n_out = n_classes)
model_name = "VGGNet"

#%%
def y_predict_model(Input_data, top_k=5):
    """
    Generates the predictions of the model over the input data, and outputs the top softmax probabilities.
        Parameters:
            X_data: Input data.
            top_k (Default = 5): The number of top softmax probabilities to be generated.
    """
    num_examples = len(Input_data)
    y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
    y_prob = np.zeros((num_examples, top_k))
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(DIR,"VGGNet.meta"))
        saver.restore(sess, os.path.join(DIR, "VGGNet"))
        graph = tf.get_default_graph()
        n_out = graph.get_tensor_by_name("n_out:0")
        y_prob, y_pred = sess.run(n_out,feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})
    return y_prob, y_pred

#%%
    
y_prob, y_pred = y_predict_model(new_test_images_preprocessed)

test_accuracy = 0
for i in enumerate(new_test_images_preprocessed):
    accu = new_IDs[i[0]] == np.asarray(y_pred[i[0]])[0]
    if accu == True:
        test_accuracy += (1/len(new_test_images))
print("New Images Test Accuracy = {:.1f}%".format(test_accuracy*100))

#%%
plt.figure(figsize=(15, 16))
for i in range(len(new_test_images_preprocessed)):
    plt.subplot(len(new_test_images_preprocessed), 2, 2*i+1)
    plt.imshow(new_test_images[i]) 
    plt.title(signs[y_pred[i][0]])
    plt.axis('off')
    plt.subplot(len(new_test_images_preprocessed), 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), y_prob[i, :])
    labels = [signs[j] for j in y_pred[i]]
    plt.yticks(np.arange(1, 6, 1), labels)
#plt.show()


# As we can notice from the top 5 softmax probabilities, the model has very high confidence (100%) when it comes to predict simple signs, like the "Stop" and the "No entry" sign, and even high confidence when predicting simple triangular signs in a very clear image, like the "Yield" sign.
# 
# On the other hand, the model's confidence slightly reduces with more complex triangular sign in a "pretty noisy" image, in the "Pedestrian" sign image, we have a triangular sign with a shape inside it, and the images copyrights adds some noise to the image, the model was able to predict the true class, but with 80% confidence.
# 
# And in the "Speed limit" sign, we can observe that the model accurately predicted that it's a "Speed limit" sign, but was somehow confused between the different speed limits. However, it predicted the true class at the end.
# 
# The VGGNet model was able to predict the right class for each of the 5 new test images. Test Accuracy = 100.0%.
# In all cases, the model was very certain (80% - 100%).

# ---
# 
# ## Conclusion
# 
# Using VGGNet, we've been able to reach a very high accuracy rate. We can observe that the models saturate after nearly 10 epochs, so we can save some computational resources and reduce the number of epochs to 10.
# We can also try other preprocessing techniques to further improve the model's accuracy..
# We can further improve on the model using hierarchical CNNs to first identify broader groups (like speed signs) and then have CNNs to classify finer features (such as the actual speed limit)
# This model will only work on input examples where the traffic signs are centered in the middle of the image. It doesn't have the capability to detect signs in the image corners.
    
#%%
save_path = "./cropped/"
#Path of video file to be read
video_read_path='speed1.mp4'

#Path of video file to be written
video_write_path='speed1.avi'

#Window Name
window_name='Input Video'

#Escape ASCII Keycode
esc_keycode=27

#Create an object of VideoCapture class to read video file
video_read = cv2.VideoCapture(video_read_path)
    # Check if video file is loaded successfully
if (video_read.isOpened()== True):
    #Frames per second in videofile. get method in VideoCapture class.
    fps = video_read.get(cv2.CAP_PROP_FPS)
    #Width and height of frames in video file
    size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #Create an object of VideoWriter class to write video file.
    #cv2.CV_FOURCC('I','4','2','0') = uncompressed YUV, 4:2:0 chroma subsampled. (.avi)
    #cv2.CV_FOURCC('P','I','M','1') = MPEG-1(.avi)
    #cv2.CV_FOURCC('M','J','P','G') = motion-JPEG(.avi)
    #cv2.CV_FOURCC('T','H','E','O') = Ogg-Vorbis(.ogv)
    #cv2.CV_FOURCC('F','L','V','1') = Flash video (.flv)
    #cv2.CV_FOURCC('M','P','4','V') = MPEG encoding (.mp4)
    #Also this form is too valid cv2.VideoWriter_fourcc(*'MJPG')
    #video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    #Set display frame rate
    display_rate = (int) (1/fps * 1000)
    #Create a Window
    #cv2.WINDOW_NORMAL = Enables window to resize.
    #cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    counter = 0
    #Read first frame from video. Return Boolean value if it succesfully reads the frame in state and captured frame in cap_frame
    state, img = video_read.read()
    #Loop untill all frames from video file are read
    imageC = 1
    xp,yp=0,0
    while state:

        #My Code

        counter+=1
        output = img.copy()
        output2 = img.copy()
        output3 = img.copy()
        # img = cv2.GaussianBlur(frame,(5,5),0)

        # MSER
        """""
        mser = cv2.MSER_create()
        regions = mser.detectRegions(img)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        cv2.imshow('img', vis)
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', hsv)

        # range
        lower_red_1 = np.array([0, 50, 50])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 50, 50])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        mask = mask1 + mask2

        red = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('res1', red)

        gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('res2', gray)

        # blur = cv2.GaussianBlur(gray,(3,3),0)
        # cv2.imshow('res3', blur)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)

        close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('res3', close)
        median = cv2.medianBlur(close, 3)

        # cv2.imshow("i",np.hstack([gray,median,close]))
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharp = cv2.filter2D(median, -1, kernel)
        # cv2.imshow('res5', sharp)

        im2, contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) & (area > 130)):
                contour_list.append(contour)

        # smoothening of contours
        '''
        smoothened = []
        for contour in contour_list:
            x, y = contour.T
            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), 25)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert it back to numpy format for opencv to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
    
    
        cv2.drawContours(output, smoothened, -1, (0, 255, 0), 3)
        cv2.fillPoly(output, pts=smoothened, color=(0, 255, 0))
        '''
        # Getting circles

        threshold = 0.50
        circle_list = []
        for contour in contour_list:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            metric = 4 * 3.14 * area / pow(perimeter, 2)
            if (metric > threshold):
                circle_list.append(contour)
        #print(circle_list)
        # cv2.drawContours(output2, circle_list, -1, (0, 255, 0), 3)
        # cv2.fillPoly(output2, pts=circle_list, color=(0, 255, 0))

        # Drawing Rectangle
        '''
        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []
    
        height, width, _ = img.shape
        min_x, min_y = width, height
        max_x = max_y = 0
    
        # computes the bounding box for the contour, and draws it on the frame,
        for contour, hier in zip(circle_list, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            if w > 80 and h > 80:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
        if max_x - min_x > 0 and max_y - min_y > 0:
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
        '''
        xm, ym, wm, hm, = 0, 0, 0, 0,
        for contour in circle_list:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            #print("m = ", [xm, ym, wm, hm])
            #print([x, y, w, h])
            if(xp==x and yp==y):
                continue
            elif (xm == 0):
                xm, ym, wm, hm = x, y, w, h
            else:
                if ((abs(xm - x) < 30) and (abs(ym - y) < 30) and (abs(wm - w)>3) and (abs(hm -h)>3)):
                    # xmean, ymean = (xm+x)//2,(ym+y)//2
                    # wmax, hmax = max(wm,w),max(hm,h)
                    if (wm > w):
                        crop = img[(ym):(ym + hm), (xm):(xm + wm)].copy()
                        xp,yp = xm,ym
                    else:
                        crop = img[(y):(y + h), (x):(x + w)].copy()
                        xp, yp = x, y
                    cv2.imwrite(os.path.join(save_path, str(imageC) + ".jpg"), crop)
                    print("IMAGE SAVED FOR CLASSIFICATION ", str(imageC))
                    imageC += 1
            xm, ym, wm, hm = x, y, w, h
            # draw rectangle around contour on original image
            cv2.rectangle(output3, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Write frame
        # if((length/counter)%100 == 0):
        print(counter)

        # Display frame
        cv2.imshow(window_name,output3)
        #Write method from VideoWriter. This writes frame to video file
        video_write.write(output3)
        #Read next frame from video
        state, img = video_read.read()
        #Check if any key is pressed.
        k = cv2.waitKey(display_rate)
        #Check if ESC key is pressed. ASCII Keycode of ESC=27
        if k == esc_keycode:
            #Destroy Window
            cv2.destroyWindow(window_name)
            break
    #Closes Video file
    video_read.release()
    video_write.release()
else:
    print("Error opening video stream or file")