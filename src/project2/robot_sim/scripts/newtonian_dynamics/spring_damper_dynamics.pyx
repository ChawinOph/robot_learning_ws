from SpringDamperDynamics cimport SpringDamperDynamics
import numpy as np
# cimport numpy as np
# cython wrapper class
cdef class SpringDamperDynamicsWrapper:
	cdef SpringDamperDynamics* dynamics 

	def __cinit__(self, double m, double k, double b):
		self.dynamics = new SpringDamperDynamics(m, k, b)

	def __init__(self, m, k, b):
		pass

	def __dealloc__(self):
		del self.dynamics

	def compute_xd(self, x, u):
		xd = np.array(self.dynamics.compute_xd(x.reshape(-1), u.reshape(-1))).reshape(-1,1)
		return xd
