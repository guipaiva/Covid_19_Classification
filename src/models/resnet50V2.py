import os

from tensorflow.keras import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from base.base_model import BaseModel
from tensorflow.keras.layers import Dense, Lambda, GlobalAveragePooling2D, BatchNormalization, Dropout



class ResNet50V2(BaseModel):
	def __init__(self, im_shape, transfer_learn, classes):
		name = 'ResNet50V2'
		super(ResNet50V2, self).__init__(name, im_shape, transfer_learn)
		self.classes = classes
		self.activation_function = 'sigmoid' if classes == 1 else 'softmax'
		self.weights = 'imagenet' if self.transfer_learn else None
		self.layers_trainable = False if self.transfer_learn else True
		self.loss = 'binary_crossentropy' if classes == 1 else 'categorical_crossentropy'

		self.build_model()

	def build_model(self):
		print('Building {}...'.format(self.name))
		self.model = tf.keras.Sequential()
		self.model.add(
			Lambda(preprocess_input, input_shape=self.im_shape, name='preproc'))

		resnet = applications.ResNet50V2(
			include_top=False,
			input_shape=self.im_shape,
			weights=self.weights
		)

		resnet.trainable = self.layers_trainable

		self.model.add(resnet)

		# FC Layers
		self.model.add(GlobalAveragePooling2D(name='avg_pool'))
		self.model.add(
			Dense(
				units=self.classes,
				activation=self.activation_function,
				name='predictions'
			),
		)

		METRICS = [
            metrics.TruePositives(name='tp'),
            metrics.FalsePositives(name='fp'),
            metrics.TrueNegatives(name='tn'),
            metrics.FalseNegatives(name='fn'), 
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
            metrics.AUC(name='prc', curve='PR')
        ]

		self.model.compile(
			optimizer='adam',
			loss=self.loss,
			metrics= METRICS
		)

		print('Model built')
