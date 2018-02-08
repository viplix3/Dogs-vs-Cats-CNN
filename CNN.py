from random import shuffle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm

# Defining some required parameters
# Use sample_train for training and sample_valid for testing to get lower execution time

train_dir = 'DataSet/train'
test_dir = 'DataSet/sample_train'

image_size = 50
lr = 1e-3
Model_Name = 'cats-vs-dogs-conv'


def create_label(image_name):
    """Creating one-hot-encoded vector from images"""
    img_name = image_name.split('.')[-3]
    if img_name == 'cat':
        return np.array([1, 0]) # 1 represents it's a cat
    elif img_name == 'dog':
        return np.array([0, 1])


def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        path = os.path.join(train_dir, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Making the image black and white
        img_data = cv2.resize(img_data, (image_size, image_size))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data 


def create_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (image_size, image_size))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save('testing_data.npy', testing_data)
    return testing_data


# Creating training and testing data if it isn't already done
train_data = create_training_data()
test_data = create_testing_data()

# If dataset has already been created you can import it using
# train_data = np.load('training_data.npy')
# test_data = np.load('testing_data.npy')

train = train_data[:-500]
test = test_data[::]
X_train = np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, image_size, image_size, 1)
y_test = [i[1] for i in test]

# Actually building the model
tf.reset_default_graph()
convnet = input_data(shape = [None, image_size, image_size, 1], name = 'input')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 100, activation='relu')
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy')

# Just some stuff that has to be done for tensorboard
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)


# This is the code that fits the data into the model
model.fit(X_train, y_train, n_epoch=10, validation_set=(X_test, y_test), snapshot_step=500, show_metric=True, run_id=Model_Name)

fig = plt.figure(figsize=(16,12))

for num, data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num + 1)
    orig = img_data
    data = img_data.reshape(image_size, image_size, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
