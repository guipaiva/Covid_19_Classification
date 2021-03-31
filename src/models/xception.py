import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import tensorflow.keras.applications as applications

class Xception(BaseModel):
    def __init__(self, im_shape):
        name = 'Xception'
        super(Xception, self).__init__(im_shape, name)
        try:
            if self.transfer_learn:
                assert len(
                    im_shape) == 3 and im_shape[0] >= 71 and im_shape[1] >= 71 and im_shape[2] == 3
            else:
                assert im_shape == (299, 299, 3)
        except AssertionError:
            print(
                'Error: Image required to have 3 channels and with and height should not be smaller than 71')
            exit(0)

        print('Building {}...'.format(self.name))
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
            self.model.add(Dense(3, activation = 'sigmoid'))
        else:
            self.model = applications.Xception(
                classes = 2,
                classifier_activation = 'sigmoid' 
            )

        self.model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        print(self.model.summary())