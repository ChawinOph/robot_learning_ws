#ifndef SPRING_DAMPER_DYNAMICS_H
#define SPRING_DAMPER_DYNAMICS_H

#include <vector>
#include "Dynamics.h"

class SpringDamperDynamics : public Dynamics
{
public:
    double m;
    double k;
    double b;
    SpringDamperDynamics(double m, double k, double b);
	std::vector<double> compute_xd(std::vector<double> x, std::vector<double> u);
};


#endif



