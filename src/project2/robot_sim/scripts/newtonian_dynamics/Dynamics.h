#ifndef DYNAMICS_H
#define DYNAMICS_H
#include <vector>

/*  

    x       - state
    u       - action
    xd      - time derivative of state
    dt      - time step

    dynamics: xd = f(x) at t
    advance: x_(t+1) from x_(t) 
    
*/ 

class Dynamics
{   

public:
    virtual ~Dynamics();
    virtual std::vector<double> compute_xd(std::vector<double> x, std::vector<double> u) = 0; 
    // compute_xd is a pure virtual function and does not have an declaration
    virtual std::vector<double> advance(std::vector<double> x, std::vector<double> u, double dt);

};


#endif