import tensorflow as tf
import tensorflow.keras.applications as applications
from base.base_model import BaseModel
from tensorflow.keras.layers import Flatten, Dense, Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResNet50V2(BaseModel):
    def __init__(self, im_shape, transfer_learn, classes):
        name = 'ResNet50'
        super(ResNet50V2, self).__init__(name, im_shape, classes, transfer_learn)
        
        self.build_model()
        
    def build_model(self):
        if self.transfer_learn:
            self.model = tf.keras.Sequential()
            resnet = applications.ResNet50V2(
                include_top=False,
                input_shape=self.im_shape
            )
            for layer in resnet.layers:
                layer.trainable = False
            self.model.add(resnet)

            # FC Layers
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(.5))

            # Classes
            if(self.classes == 2)
                self.model.add(Dense(self.classes, activation='sigmoid'))
            else:
                self.model.add(Dense(self.classes, activation = 'softmax'))
        else:
            self.model = applications.ResNet50V2(
				include_top = True,
				weights = None,
				classes = self.classes
            )

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print('Model built')