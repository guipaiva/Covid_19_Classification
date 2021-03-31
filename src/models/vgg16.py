import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VGG16(BaseModel):
    def __init__(self, im_shape):
        name = 'VGG16'
        super(VGG16, self).__init__(im_shape, name)
        try:
            if self.transfer_learn:
                assert len(
                    im_shape) == 3 and im_shape[0] >= 32 and im_shape[1] >= 32 and im_shape[2] == 3
            else:
                assert im_shape == (224, 224, 3)
        except AssertionError:
            print(
                'Error: Image required to have 3 channels and with and height should not be smaller than 32')
            exit(0)

        print('Building {}...'.format(self.name))
        self.build_model()

    def build_model(self):
		if self.transfer_learn:
			self.model = tf.keras.Sequential()
			vgg = applications.VGG16(include_top=False, input_shape=self.im_shape)
			for layer in vgg.layers:
				layer.trainable = False

			self.model.add(vgg)
			self.model.add(Flatten())
			self.model.add(Dense(1024, activation='relu'))
			self.model.add(Dropout(.5))
			self.model.add(Dense(3, activation='sigmoid'))
		else:
			self.model = applications.VGG16(
				classes = 2,
				classifier_activation = 'sigmoid' 
			)

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
