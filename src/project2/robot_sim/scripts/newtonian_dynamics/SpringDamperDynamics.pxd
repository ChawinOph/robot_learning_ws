from libcpp.vector cimport vector

cdef extern from "SpringDamperDynamics.h":
	cdef cppclass SpringDamperDynamics: 
		double m, k, b
		SpringDamperDynamics() except +
		SpringDamperDynamics(double, double, double) except +
		vector[double] compute_xd(vector[double], vector[double])