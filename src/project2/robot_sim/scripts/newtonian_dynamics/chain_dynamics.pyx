from ChainDynamics cimport ArmDynamics
from ChainDynamics cimport SnakeDynamics
import numpy as np

# ArmDyanmicsWrapper and SnakeDynamicsWrapper are the cython wrapper classes that 
# wrap C++ ArmDynamics and SnakeDynamics. It may be possible to make them inherit 
# from a DynamicsWrapper class to avoid code dublication. Non-trivial yet.

cdef class ArmDynamicsWrapper:
	cdef ArmDynamics* dynamics
	cdef int num_links
	cdef double link_mass
	cdef double link_length
	cdef double torques_max
	cdef double coeff_viscous_joint
	cdef double coeff_viscous_link_forward
	cdef double coeff_viscous_link_sideways
	cdef double coeff_dry_link_forward
	cdef double coeff_dry_link_sideways
	cdef double gravity_y
	cdef double gravity_z

	def __cinit__(self, int num_links, double link_mass, double link_length, double torques_max,
	    double coeff_viscous_joint, double coeff_viscous_link_forward, double coeff_viscous_link_sideways,
		double coeff_dry_link_forward, double coeff_dry_link_sideways, double gravity_y, double gravity_z):
		
		self.dynamics = new ArmDynamics(num_links, link_mass, link_length, torques_max, coeff_viscous_joint,
			coeff_viscous_link_forward, coeff_viscous_link_sideways, coeff_dry_link_forward, coeff_dry_link_sideways,
			gravity_y, gravity_z)

		# cache constructor parameters for deepcopy ( __reduce__() )
		self.num_links = num_links
		self.link_mass = link_mass
		self.link_length = link_length
		self.torques_max = torques_max
		self.coeff_viscous_joint = coeff_viscous_joint
		self.coeff_viscous_link_forward = coeff_viscous_link_forward
		self.coeff_viscous_link_sideways = coeff_viscous_link_sideways
		self.coeff_dry_link_forward = coeff_dry_link_forward
		self.coeff_dry_link_sideways = coeff_dry_link_sideways
		self.gravity_y = gravity_y
		self.gravity_z = gravity_z

	def __init__(self, *args, **kwargs):
		pass

	def __dealloc__(self):
		if not self.dynamics == NULL:
			del self.dynamics

	def compute_xd(self, x, u):
		x = x.reshape(-1)
		u = u.reshape(-1)
		xd = np.array(self.dynamics.compute_xd(x, u))
		xd = xd.reshape(-1,1)
		return xd

	def advance(self, x, u, dt):
		x = x.reshape(-1)
		u = u.reshape(-1)
		xnew = np.array(self.dynamics.advance(x, u, dt))
		xnew = xnew.reshape(-1,1)
		return xnew

	def compute_energy(self, x):
		x = x.reshape(-1)
		energy = np.array(self.dynamics.compute_energy(x))
		return energy

	def __reduce__(self):
		return (self.__class__, (self.num_links, self.link_mass, self.link_length, self.torques_max,
			self.coeff_viscous_joint, self.coeff_viscous_link_forward, self.coeff_viscous_link_sideways,
			self.coeff_dry_link_forward, self.coeff_dry_link_sideways, self.gravity_y, self.gravity_z))

cdef class SnakeDynamicsWrapper:
	cdef SnakeDynamics* dynamics
	cdef int num_links
	cdef double link_mass
	cdef double link_length
	cdef double torques_max
	cdef double coeff_viscous_joint
	cdef double coeff_viscous_link_forward
	cdef double coeff_viscous_link_sideways
	cdef double coeff_dry_link_forward
	cdef double coeff_dry_link_sideways
	cdef double gravity_y
	cdef double gravity_z 

	def __cinit__(self, int num_links, double link_mass, double link_length, double torques_max,
		double coeff_viscous_joint, double coeff_viscous_link_forward, double coeff_viscous_link_sideways,
		double coeff_dry_link_forward, double coeff_dry_link_sideways, double gravity_y, double gravity_z):

		self.dynamics = new SnakeDynamics(num_links, link_mass, link_length, torques_max, coeff_viscous_joint,
			coeff_viscous_link_forward, coeff_viscous_link_sideways, coeff_dry_link_forward, coeff_dry_link_sideways,
			gravity_y, gravity_z)

		# cache constructor parameters for deepcopy ( __reduce__() )
		self.num_links = num_links
		self.link_mass = link_mass
		self.link_length = link_length
		self.torques_max = torques_max
		self.coeff_viscous_joint = coeff_viscous_joint
		self.coeff_viscous_link_forward = coeff_viscous_link_forward
		self.coeff_viscous_link_sideways = coeff_viscous_link_sideways
		self.coeff_dry_link_forward = coeff_dry_link_forward
		self.coeff_dry_link_sideways = coeff_dry_link_sideways
		self.gravity_y = gravity_y
		self.gravity_z = gravity_z

	def __init__(self, *args, **kwargs):
		pass

	def __dealloc__(self):
		if not self.dynamics == NULL:
			del self.dynamics

	def compute_xd(self, x, u):
		x = x.reshape(-1)
		u = u.reshape(-1)
		xd = np.array(self.dynamics.compute_xd(x, u))
		xd = xd.reshape(-1,1)
		return xd

	def advance(self, x, u, dt):
		x = x.reshape(-1)
		u = u.reshape(-1)
		xnew = np.array(self.dynamics.advance(x, u, dt))
		xnew = xnew.reshape(-1,1)
		return xnew

	def compute_energy(self, x):
		x = x.reshape(-1)
		energy = np.array(self.dynamics.compute_energy(x))
		return energy

	def __reduce__(self):
		return (self.__class__, (self.num_links, self.link_mass, self.link_length, self.torques_max,
			self.coeff_viscous_joint, self.coeff_viscous_link_forward, self.coeff_viscous_link_sideways,
			self.coeff_dry_link_forward, self.coeff_dry_link_sideways, self.gravity_y, self.gravity_z))