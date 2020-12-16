from base.base_trainer import BaseTrainer
import datetime
import tensorflow as tf

class SimpleTrainer(BaseTrainer):
	def __init__(self, model, data, epochs):
		super(SimpleTrainer, self).__init__(model, data, epochs)
		self.log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
		self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)]
		self.accuracy = []
		self.loss = []
		self.val_accuracy = []
		self.val_loss = []

	def train(self):
		history = self.model.fit(
			self.data["train"],
			validation_data = self.data["validation"],
			callbacks = self.callbacks,
			epochs = self.epochs,
			verbose = 2
		)
		self.loss.extend(history.history['loss'])
		self.accuracy.extend(history.history['accuracy'])
		self.val_loss.extend(history.history['val_loss'])
		self.val_accuracy.extend(history.history['val_accuracy'])
		