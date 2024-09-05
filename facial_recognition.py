
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from shutil import unpack_archive
from subprocess import check_output

batchsize = 16
img_height = 512
img_width = 512


train_path = os.path.abspath("./train_ds")
test_path = os.path.abspath("./test_ds")
valid_path = os.path.abspath("./valid_ds")


#pulling training/validation/testing data from directories
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(img_height, img_width),
    batch_size=batchsize,
    label_mode="categorical"
)
class_names = train_data.class_names

val_data = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=(img_height, img_width),
    batch_size=batchsize,
    label_mode="categorical"

)
class_names = train_data.class_names

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(img_height, img_width),
    batch_size=batchsize,
    label_mode="categorical"

)

class_number=len(train_data.class_names)

model = tf.keras.Sequential([
    
    #normalize the pixels in images to be between 0-1    
    tf.keras.layers.Rescaling(scale=1./255.),
    tf.keras.layers.Conv2D(32, 3, activation='relu',padding="valid"),
    tf.keras.layers.Dropout(0.3),
    #tried batchnormalization vs dropout, batch normalization had little to no impact whereas dropout greatly helped
    ##tried average pooling, same issue as l1 regualizers, caused too many missing values and prevented any progress in training/validation loss
    
    #i've tried adding more layers to the model, this has helped slighlty but the increase in parameters has decreased the models effectiveness for product
    #due to high memory and computational time come runtime
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(), 
    
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(), 
    
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    
    
    #units massively increases parameters and causes memory problems and greater levels, but has shown good increases in accuracy    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    #softmax of number of individuals in the class
    tf.keras.layers.Dense(class_number, activation="softmax")  
        
])


model.compile(
    #have been messing around with different learning rates, hit or miss on different values, ranges between 0.001-0.01 work best
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)



epochs = 5
history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs,
  batch_size=batchsize
)

#test model
model.evaluate(test_data)
model.save('model.h5')

###all below is simple plot for accuracy vs loss 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

plt.savefig('accuracy_loss')