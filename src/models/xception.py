import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from base.base_model import BaseModel
import tensorflow.keras.applications as applications

class transfer_learning_Xception(BaseModel):
    def __init__(self, im_shape):
        name = 'Xception'
        super(transfer_learning_Xception, self).__init__(im_shape, name)
        try: 
            assert im_shape == (299, 299, 3)
        except AssertionError: 
            print('Error: Xception requires image shape to be 299x299x3')	
            exit(0)

        print('Building...')

    def build_model(self):
        self.model = tf.keras.Sequential()
        xcp = applications.Xception(include_top= False, input_shape = self.im_shape)

        for layer in xcp.layers:
            layer.trainable = False
        self.model.add(xcp)

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