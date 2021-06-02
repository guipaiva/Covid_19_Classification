import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.applications.densenet import preprocess_input
from base.base_model import BaseModel
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DenseNet121(BaseModel):
    def __init__(self, im_shape, classes, transfer_learn):
        name = 'DenseNet121'
        super(DenseNet121, self).__init__(name, im_shape, transfer_learn)
        self.classes = classes
        self.activation_function = 'sigmoid' if classes == 2 else 'softmax'
        self.weights = 'imagenet' if self.transfer_learn else None
        self.layers_trainable = False if self.transfer_learn else True
        self.loss = 'binary_crossentropy' if classes == 2 else 'categorical_crossentropy'

        self.build_model()

    def build_model(self):
        print('Building {}...'.format(self.name))
        self.model = tf.keras.Sequential()
        self.model.add(
            Lambda(preprocess_input, input_shape=self.im_shape, name='preproc'))

        densenet = applications.DenseNet121(
            include_top=False,
            input_shape=self.im_shape,
            weights=self.weights
        )

        for layer in densenet.layers:
            layer.trainable = self.layers_trainable
        self.model.add(densenet)

        # FC Layers
        self.model.add(GlobalAveragePooling2D(name='avg_pool'))
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
