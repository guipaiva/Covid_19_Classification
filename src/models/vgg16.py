import tensorflow as tf
import tensorflow.keras.applications as applications
#pylint: disable=import-error
from base.base_model import BaseModel
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Flatten, Lambda)


class VGG16(BaseModel):
    def __init__(self, im_shape, transfer_learn, classes, metrics):
        name = 'VGG16'
        super(VGG16, self).__init__(name, im_shape, transfer_learn, classes, metrics)
        self.build_model()

    def build_model(self):
        print('Building {}...'.format(self.name))
        self.model = tf.keras.Sequential()
        self.model.add(
            Lambda(preprocess_input, input_shape=self.im_shape, name='preproc'))

        base_vgg = applications.VGG16(
            include_top=False,
            input_shape=self.im_shape,
            weights=self.weights
        )

        base_vgg.trainable = self.layers_trainable

        self.model.add(base_vgg)

        # Base top layers
        self.model.add(Flatten(name='Flatten'))
        self.model.add(
            Dense(
                units=4096,
                activation='relu',
                name='VGG_FC1'
            )
        )
        self.model.add(
            Dense(
                units=4096,
                activation='relu',
                name='VGG_FC2'
            )
        )
        
        # More FC Layers
        self.model.add(
            Dense(
                units=512,
                activation='relu',
                name='Dense1'
            )
        )
        self.model.add(
            Dropout(rate = 0.5)
        )
        self.model.add(
            Dense(
                units=128,
                activation='relu',
                name='Dense2'
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
            metrics=self.metrics
        )

        print('Model built')
