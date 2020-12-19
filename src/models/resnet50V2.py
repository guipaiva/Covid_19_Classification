import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import tensorflow.keras.applications as applications

class transfer_learning_Resnet50V2(BaseModel):
	def __init__(self, im_shape):
		name = 'Resnet50'
		super(transfer_learning_Resnet50V2, self).__init__(im_shape, name)
		try: 
			assert len(im_shape) == 3 and im_shape[-1] == 3
		except AssertionError: 
			print('Error: Image required to have 3 channels')
			exit(0)

		print('Building...')
		self.build_model()

	def build_model(self):
		self.model = tf.keras.Sequential()
		resnet = applications.ResNet50V2(include_top= False, input_shape = self.im_shape)

		for layer in resnet.layers:
			layer.trainable = False
		self.model.add(resnet)

		#FC Layers
		self.model.add(Flatten())
		self.model.add(Dense(512, activation = 'relu'))
		self.model.add(Dropout(.5))
		self.model.add(Dense(3, activation = 'softmax'))

		self.model.compile(
			optimizer = 'adam',
			loss = 'categorical_crossentropy',
			metrics = ['accuracy']
		)