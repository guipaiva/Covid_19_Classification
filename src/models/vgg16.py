import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.path.append('c:\\Users\\X1200WK_05\\Documents\\Guilherme\\NNs\\Utils')

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from read import Image_generator
import datetime
im_size = 224

vgg = VGG16(include_top= False, input_shape = (im_size, im_size, 3))

for layer in vgg.layers:
    layer.trainable = False

model = tf.keras.Sequential()
model.add(vgg)

#FC Layers
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(.5))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

generator = Image_generator(im_size)
train, valid = generator.generate()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)




trainer = model.fit(
        train,
        validation_data = valid,
        epochs = 8,
        verbose = 2,
        callbacks=[tensorboard_callback],
    )