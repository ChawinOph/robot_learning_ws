#include <iostream>
#include "SpringDamperDynamics.h"
using namespace std;

SpringDamperDynamics::SpringDamperDynamics(double m, double k, double b): m(m), k(k), b(b) {}

vector<double> SpringDamperDynamics::compute_xd(vector<double> xv, vector<double> uv){
	double x, v, f, a;
	x = xv[0];
	v = xv[1];
	f = uv[0];
	a = (f - k*x - b*v)/m;
	vector<double> xd{v,a};
	return xd;
};


