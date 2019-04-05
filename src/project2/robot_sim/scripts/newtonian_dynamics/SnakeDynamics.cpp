#include<iostream>
#include<vector>
#include"SnakeDynamics.h"
#include<eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;


SnakeDynamics::SnakeDynamics(uint num_links, double link_mass, double link_length, double torques_max,
		double coeff_viscous_joint, double coeff_viscous_link_forward, double coeff_viscous_link_sideways,
		double coeff_dry_link_forward, double coeff_dry_link_sideways, double gravity_y, double gravity_z):
	ChainDynamics( num_links, link_mass, link_length, torques_max, coeff_viscous_joint, coeff_viscous_link_forward,
		coeff_viscous_link_sideways, coeff_dry_link_forward, coeff_dry_link_sideways, gravity_y, gravity_z){}


vector<double> SnakeDynamics::wrap_x_in(vector<double> x_in){
	return x_in;
}

vector<double> SnakeDynamics::wrap_u_in(vector<double> u_in){
	vector<double> out(num_links_, 0.0);
	copy(u_in.begin(), u_in.end(), out.begin()+1);
	return out;
}

vector<double> SnakeDynamics::wrap_x_out(vector<double> x_out){
	return x_out;
}

void SnakeDynamics::setup_constraints(vector<double> const & q, vector<double> const & qd, vector<double> const & theta,
        vector<double> const & omega, vector<vector<double>> const & vel_com, vector<double> const & tau,
        MatrixXd& CL, VectorXd& CR){

	ChainDynamics::setup_constraints(q, qd, theta, omega, vel_com, tau, CL, CR);

	// add snake related constraints here
	// viscous friction torque on intermediate joints
	for(uint i=1; i < num_links_; i++){
		CR(idx_tau_eq(i)) += qd[i] * nu_q_;
	}
	// force at joint-0 must be zero
	CL.block<2,2>(idx_mode_eq(), idx_f(0)) = eye(2);
}


