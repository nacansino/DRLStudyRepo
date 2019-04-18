#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:01:21 2019

@author: jay
"""
from keras.models import Dropout #DROPOUT
from keras.callbacks import Callback
# Importing the Keras libraries and packages
from keras.models import Sequential #classifier
from keras.layers import Conv2D #convolution
from keras.layers import MaxPooling2D #pooling
from keras.layers import Flatten #flattening the pooled features to a long feature vector
from keras.layers import Dense #for hidden layers

#callback class
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'


#memo: VGG16 architecture (pre-trained classifiers)

#set file input size
input_size=(128, 128)

# Initialising the CNN
classifier = Sequential()


# Step 1 - Convolution
# Contrary to traditional ANN, 
# the first layer that we will going to add is the convolutional layer 
# to obtain feature maps.
# filter: # of feature detectors
# kernel_size: height and width of the 2d convolution window
# input_shape:  shape of your input image (3d array for colored images, 2d arrays for b/w images)
#               For TensorFlow backend, (dimx, dimy, colors)
# activation: activation function (use rectifier activation function)
# Idea: Features are also trained on the fly 
#       (aka together with the activations of the nodes in the hidden layers)
classifier.add(Conv2D(32, (3, 3), input_shape = (*input_size, 3), activation = 'relu'))

# Step 2 - Pooling
# Used to reduce the size of each feature map (the number of w/c is indicated above)
# If size of feature map is odd, then new size is (n/2+1)x(n/2+1). If even, then (n/2 x n/2).
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# this will increase the accuracy of the classifier
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Add classical layers
# ROT: choose a number between the number of input nodes and output nodes
# Note that "units" means the output of the layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
# use sigmoid fxn for binary output (one class)
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# adding another dense layer may also improve the accuracy

# Compiling the CNN
# Same process as the previous section - ANN
# Loss fxn: binary_crossentropy for binary output. use categorial cross entropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# Preprocessing images to avoid overfitting
# What this class essentially does is it creates multiple batches of versions of images
# that will serve as additional data for our neural network.
# for images, the class generates augmented data by rotating them, flipping them, or shifting them.
from keras.preprocessing.image import ImageDataGenerator

# Example of using .flow_from_directory(directory):

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# create training set
# set expected images and batch size
# Note that increasing target_size may also increase accuracy (utilizing more pixels from the input data)
# but will also increase your training time.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = input_size,
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# create test set
# similar as above
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = input_size,
                                            batch_size = 32,
                                            class_mode = 'binary')
# Create a loss history
history = LossHistory()

# fit generator
# steps per epoch: essentially # of training set images
# (Since this new API accepts the number of steps per BATCH SIZE instead of datapoints,
# we use (#datapoints)/batch_size)
classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32),
                         use_multiprocessing=True,
                         workers=4,
                          callbacks=[history])

# Save model
model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, '../loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)
