import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.applications.resnet import preprocess_input
from base.base_model import BaseModel
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResNet50(BaseModel):
	def __init__(self, im_shape, transfer_learn, classes):
		name = 'ResNet50'
		super(ResNet50, self).__init__(name, im_shape, classes, transfer_learn)
		self.activation_function = 'sigmoid' if classes == 2 else 'softmax'
		self.weights = 'imagenet' if self.transfer_learn else None
		self.layers_trainable = False if self.transfer_learn else True
		self.loss = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'
		self.build_model()

	def build_model(self):

		self.model = tf.keras.Sequential()

		self.model.add(Lambda(preprocess_input, input_shape = (224,224,3)))

		resnet = applications.ResNet50(
			include_top=False,
			input_shape=self.im_shape,
			weights= self.weights
		)

		for layer in resnet.layers:
			layer.trainable = self.layers_trainable

		self.model.add(resnet)

		# FC Layers
		self.model.add(GlobalAveragePooling2D())
		self.model.add(
			Dense(
				units=self.classes,
				activation=self.activation_function,
				name='predictions'
			),
		)

		self.model.compile(
			optimizer='adam',
			loss=self.loss,
			metrics=['accuracy']
		)

		print('Model built')
