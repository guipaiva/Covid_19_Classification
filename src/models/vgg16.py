import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from base_model import BaseModel
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow.keras.applications as applications

class transfer_learning_VGG16(BaseModel):
	def __init__(self, im_shape):
		super(transfer_learning_VGG16, self).__init__(im_shape)
		try: 
			assert im_shape == (224, 224, 3)
		except AssertionError: 
			print('Error: VGG16 requires image shape to be 224x224x3')	
			exit(0)
		print("Building...")
		self.build_model()
	
	def build_model(self):
		self.model = tf.keras.Sequential()
		vgg = applications.VGG16(include_top= False, input_shape = (self.im_shape))
		for layer in vgg.layers:
			layer.trainable = False
		
		self.model.add(vgg)
		self.model.add(Flatten())
		self.model.add(Dense(1024, activation = 'relu'))
		self.model.add(Dropout(.5))
		self.model.add(Dense(3, activation = 'softmax'))

		self.model.compile(
			optimizer = 'adam',
			loss = 'sparse_categorical_crossentropy',
			metrics = ['accuracy']
		)