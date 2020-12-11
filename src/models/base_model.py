class BaseModel(object):
	def __init__(self, im_shape):
		self.model = None
		self.im_shape = im_shape
	
	def build_model(self):
		raise NotImplementedError
