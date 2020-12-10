class BaseLoader(object):
	def __init__(self, directory, im_size, batch_size=32):
		self.im_size = im_size
		self.batch_size = batch_size
		self.directory = directory

	def get_train_ds(self):
		raise NotImplementedError

	def get_validation_ds(self):
		raise NotImplementedError
