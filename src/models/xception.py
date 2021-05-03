import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import tensorflow.keras.applications as applications

class Xception(BaseModel):
    def __init__(self, im_shape, classes, transfer_learn):
        name = 'Xception'
        super(Xception, self).__init__(name, im_shape, classes, transfer_learn)
        self.build_model()

    def build_model(self):
        if self.transfer_learn:
            self.model = tf.keras.Sequential()
            xcp = applications.Xception(include_top= False, input_shape = self.im_shape)

            for layer in xcp.layers:
                layer.trainable = False
            self.model.add(xcp)

            #FC Layers
            self.model.add(Flatten())
            self.model.add(Dense(512, activation = 'relu'))
            self.model.add(Dropout(.5))
            #Prediction
            if self.classes == 2:
                self.model.add(Dense(2, activation = 'sigmoid'))
            else:
                self.model.add(Dense(self.classes, activation = 'softmax'))
        else:
            self.model = applications.Xception(
                include_top = True,
                weights = None,
                classes = self.classes
            )

        self.model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        print(self.model.summary())