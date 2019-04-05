#ifndef SNAKE_DYNAMICS_H
#define SNAKE_DYNAMICS_H

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "ChainDynamics.h"


class SnakeDynamics: public ChainDynamics{

public:

	SnakeDynamics(uint num_links, double link_mass, double link_length, double torques_max, double coeff_viscous_joint, 
		double coeff_viscous_link_forward, 
		double coeff_viscous_link_sideways, 
	    double coeff_dry_link_forward, 
	    double coeff_dry_link_sideways,
	    double gravity_y,
	    double gravity_z);

    std::vector<double> wrap_x_in(std::vector<double> x); 
    std::vector<double> wrap_u_in(std::vector<double> u);
    std::vector<double> wrap_x_out(std::vector<double> x);

    void setup_constraints(std::vector<double> const & q, std::vector<double> const & qd,
            std::vector<double> const & theta, std::vector<double> const & omega,
            std::vector<std::vector<double>> const & vel_com, std::vector<double> const & tau,
            Eigen::MatrixXd& CL, Eigen::VectorXd& CR);

	uint get_state_dim() {return  2*num_links_+4;}
	uint get_action_dim() {return  num_links_-1;}
};

#endif