import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from base.base_model import BaseModel
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout



class SimpleCNN(BaseModel):
	def __init__(self, im_shape):
		super(SimpleCNN, self).__init__(im_shape)
		self.build_model()
	
	def build_model(self):
		self.model = tf.keras.Sequential()

		#Layer 1
		self.model.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3), strides = (2,2), input_shape = (self.im_shape)))
		self.model.add(MaxPool2D(pool_size = (2,2)))

		#layer 2
		self.model.add(Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3), strides = (2,2)))
		self.model.add(MaxPool2D(pool_size = (2,2)))

		#layer 3
		self.model.add(Conv2D(filters = 128, activation = 'relu', kernel_size = (3,3), strides = (2,2)))
		self.model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))

		#Fully Connected
		self.model.add(Flatten())
		self.model.add(Dense(32, activation = 'relu'))
		self.model.add(Dropout(.3))
		self.model.add(Dense(3, activation = 'softmax'))


		self.model.compile(
			optimizer = 'adam',
			loss = 'sparse_categorical_crossentropy',
			metrics = ['accuracy']
		)


# trainer = model.fit(train, validation_data = valid, epochs = 8, verbose = 2)