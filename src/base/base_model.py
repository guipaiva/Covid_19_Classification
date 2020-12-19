class BaseModel(object):
	def __init__(self, im_shape, name):
		self.model = None
		self.im_shape = im_shape
		self.name = name
	
	def build_model(self):
		raise NotImplementedError
