import tensorflow as tf
import tensorflow.keras.applications as applications
#pylint: disable=import-error
from base.base_model import BaseModel
from tensorflow.keras import metrics
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda


class Xception(BaseModel):
    def __init__(self, im_shape, transfer_learn, classes, metrics):
        name = 'Xception'
        super(Xception, self).__init__(name, im_shape, transfer_learn, classes, metrics)
        self.build_model()

    def build_model(self):
        print('Building {}...'.format(self.name))
        self.model = tf.keras.Sequential()
        self.model.add(Lambda(preprocess_input, input_shape=self.im_shape, name = 'preproc'))

        base_xcp = applications.Xception(
            include_top=False,
            input_shape=self.im_shape,
            weights=self.weights
        )

        base_xcp.trainable = self.layers_trainable

        self.model.add(base_xcp)

        #Top layers
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
            metrics=self.metrics
        )

        print('Model built')
