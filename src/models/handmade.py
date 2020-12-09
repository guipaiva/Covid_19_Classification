import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from read import generate

model = tf.keras.Sequential()

im_size = 256

#Layer 1
model.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3), strides = (2,2), input_shape = (im_size, im_size, 3)))
model.add(MaxPool2D(pool_size = (2,2)))

#layer 2
model.add(Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3), strides = (2,2)))
model.add(MaxPool2D(pool_size = (2,2)))

#layer 3
model.add(Conv2D(filters = 128, activation = 'relu', kernel_size = (3,3), strides = (2,2)))
model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))

#Fully Connected
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

print(model.summary())
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train, valid = generate(im_size)

trainer = model.fit(train, validation_data = valid, epochs = 8, verbose = 2)