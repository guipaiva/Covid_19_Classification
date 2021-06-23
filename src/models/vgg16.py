import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Lambda, Flatten
from base.base_model import BaseModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VGG16(BaseModel):
    def __init__(self, im_shape, transfer_learn, classes):
        name = 'VGG16'
        super(VGG16, self).__init__(name, im_shape, transfer_learn)
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

        vgg = applications.VGG16(
            include_top=False,
            input_shape=self.im_shape,
            weights=self.weights
        )

        for layer in vgg.layers:
            layer.trainable = self.layers_trainable

        self.model.add(vgg)

        # FC layers
        self.model.add(Flatten(name='Flatten'))
        self.model.add(
            Dense(
                units=4096,
                activation='relu',
                name='FC1'
            )
        )
        self.model.add(
            Dense(
                units=4096,
                activation='relu',
                name='FC2'
            )
        )
        self.model.add(
            Dense(
                units=self.classes,
                activation=self.activation_function,
                name='predictions'
            )
        )

        self.model.compile(
            optimizer='adam',
            loss=self.loss,
            metrics=['accuracy']
        )

        print('Model built')
