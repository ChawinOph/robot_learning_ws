from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           	"spring_damper_dynamics",               # the extension name
           	sources=["spring_damper_dynamics.pyx", "SpringDamperDynamics.cpp", "Dynamics.cpp"], # the Cython source and
                                                  	# additional C++ source files
           	language="c++",                        	# generate and compile C++ code
           	extra_compile_args=["-std=c++11"],
    		extra_link_args=["-std=c++11"]
      )))


setup(ext_modules = cythonize(Extension(
           	"chain_dynamics",               # the extension name
           	sources=["chain_dynamics.pyx", "Dynamics.cpp", "ChainDynamics.cpp", "ArmDynamics.cpp", "SnakeDynamics.cpp"], # the Cython source and
                                                  	# additional C++ source files
           	language="c++",                        	# generate and compile C++ code
           	extra_compile_args=["-std=c++11"],
    		extra_link_args=["-std=c++11"]
      )))