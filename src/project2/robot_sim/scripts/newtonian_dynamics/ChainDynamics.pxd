from libcpp.vector cimport vector

cdef extern from "ArmDynamics.h":
	cdef cppclass ArmDynamics: 
		ArmDynamics(int, double, double, double, double, double, double, double, double, double, double) except +
		vector[double] compute_xd(vector[double], vector[double]) except +
		vector[double] advance(vector[double], vector[double], double)
		vector[double] compute_energy(vector[double])

cdef extern from "SnakeDynamics.h":
	cdef cppclass SnakeDynamics: 
		SnakeDynamics(int, double, double, double, double, double, double, double, double, double, double) except +
		vector[double] compute_xd(vector[double], vector[double]) except +
		vector[double] advance(vector[double], vector[double], double)
		vector[double] compute_energy(vector[double])