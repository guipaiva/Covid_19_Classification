import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.xception import Xception


im_size = 299

xcp = Xception(include_top= False, input_shape = (im_size, im_size, 3))

for layer in xcp.layers:
    layer.trainable = False

model = tf.keras.Sequential()
model.add(xcp)

#FC Layers
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(.5))
model.add(Dense(3, activation = 'softmax'))

#print(model.summary())

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train, valid = generate(im_size)

trainer = model.fit(train, validation_data = valid, epochs = 8, verbose = 2)