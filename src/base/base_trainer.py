class BaseTrainer(object):
	def __init__(self, model, data, epochs):
		self.model = model
		self.data = data
		self.epochs = epochs
	def train(self):
		raise NotImplementedError