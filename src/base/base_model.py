class BaseModel(object):
	def __init__(self, im_shape, name, classes, transfer_learn):
		self.model = None
		self.im_shape = im_shape
		self.name = name
		self.classes = classes
		self.transfer_learn = transfer_learn
	
	def build_model(self):
		raise NotImplementedError
