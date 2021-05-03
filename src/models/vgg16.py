import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VGG16(BaseModel):
    def __init__(self, im_shape, classes, transfer_learn):
        name = 'VGG16'
        super(VGG16, self).__init__(name, im_shape, classes, transfer_learn)
        self.build_model()

    def build_model(self):
		if self.transfer_learn:
			self.model = tf.keras.Sequential()
			vgg = applications.VGG16(
                include_top=False, 
                input_shape=self.im_shape
            )
			for layer in vgg.layers:
				layer.trainable = False

			self.model.add(vgg)

			self.model.add(Flatten())
			self.model.add(Dense(1024, activation='relu'))
			self.model.add(Dropout(.5))
            #Prediction
            if classes == 2:
			    self.model.add(Dense(self.classes, activation='sigmoid'))
            else:
                self.model.add(Dense(self.classes, activation='softmax'))
		else:
			self.model = applications.VGG16(
				include_top = True,
				weights = None,
				classes = self.classes
			)

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
