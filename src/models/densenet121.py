import tensorflow as tf
import tensorflow.keras.applications as applications
from base.base_model import BaseModel
from tensorflow.keras.layers import Flatten, Dense, Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DenseNet121(BaseModel):
	def __init__(self, im_shape, transfer_learn, classes):
		name = 'DenseNet121'
		super(DenseNet121, self).__init__(im_shape, name, classes, transfer_learn)

		try:
			if self.transfer_learn:
				assert len(im_shape) == 3 and im_shape[0] >= 32 and im_shape[1] >= 32 and im_shape[2] == 3
			else:
				assert im_shape == (224, 224, 3)
		except AssertionError:
			if self.transfer_learn:
				print('Error: Image required to have 3 channels and with and height should not be smaller than 32')
			else:
				print('Error: Image shape without transfer learning must be 224x224x3')
			exit(0)

		print('Building {}...'.format(self.name))
		self.build_model()

	def build_model(self):
		if self.transfer_learn:
			self.model = tf.keras.Sequential()
			densenet = applications.DenseNet121(
				include_top=False, 
				input_shape=self.im_shape
			)

			for layer in densenet.layers:
				layer.trainable = False
			self.model.add(densenet)

			# FC Layers
			self.model.add(Flatten())
			self.model.add(Dense(512, activation='relu'))
			self.model.add(Dropout(.5))

			# Classes
			if self.classes == 2:
				self.model.add(Dense(self.classes, activation = 'sigmoid'))
			else:
				self.model.add(Dense(self.classes, activation='softmax'))

		else:
			self.model = applications.DenseNet121(
				include_top = True,
				weights = None,
				classes = self.classes
			)

		self.model.compile(
			optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy']
		)

		print('Model {} built'.format(self.name))
