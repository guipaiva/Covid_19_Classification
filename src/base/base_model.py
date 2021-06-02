from utils.definitions import MODELS_SHAPE

class BaseModel(object):
	def __init__(self, name, im_shape, transfer_learn):
		self.model = None
		self.name = name
		self.im_shape = im_shape
		self.transfer_learn = transfer_learn

	def build_model(self):
		raise NotImplementedError
