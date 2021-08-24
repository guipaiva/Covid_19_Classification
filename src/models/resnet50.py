import tensorflow as tf
import tensorflow.keras.applications as applications
#pylint: disable=import-error
from base.base_model import BaseModel
from tensorflow.keras import metrics
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout


class ResNet50(BaseModel):
    def __init__(self, im_shape, transfer_learn, classes, metrics):
        name = 'ResNet50'
        super(ResNet50, self).__init__(name, im_shape, transfer_learn, classes, metrics)

        self.build_model()

    def build_model(self):
        print('Building {}...'.format(self.name))
        self.model = tf.keras.Sequential()
        if not self.transfer_learn:
            self.model.add(
                Lambda(preprocess_input, input_shape=self.im_shape, name = 'preproc'))

        base_resnet = applications.ResNet50(
            include_top=False,
            input_shape=self.im_shape,
            weights=self.weights
        )

        base_resnet.trainable = self.layers_trainable

        self.model.add(base_resnet)

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
            metrics=self.metrics
        )

        print('Model built')
		