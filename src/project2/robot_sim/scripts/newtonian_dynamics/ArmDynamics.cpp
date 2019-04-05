#include<iostream>
#include<vector>
#include "ArmDynamics.h"

using namespace std;
using namespace Eigen;

ArmDynamics::ArmDynamics(uint num_links, double link_mass, double link_length, double torques_max,
		double coeff_viscous_joint, double coeff_viscous_link_forward, double coeff_viscous_link_sideways,
		double coeff_dry_link_forward, double coeff_dry_link_sideways, double gravity_y, double gravity_z):
	ChainDynamics(num_links, link_mass, link_length, torques_max, coeff_viscous_joint, coeff_viscous_link_forward,
		coeff_viscous_link_sideways, coeff_dry_link_forward, coeff_dry_link_sideways, gravity_y, gravity_z){}

vector<double> ArmDynamics::wrap_x_in(vector<double> x_in){
	vector<double> x(2*num_links_+4, 0.0); // vector with zeros
	copy(x_in.begin(), x_in.begin()+num_links_, x.begin()+2);
	copy(x_in.begin()+num_links_, x_in.end(), x.begin()+num_links_+4);
	return x;
}

vector<double> ArmDynamics::wrap_u_in(vector<double> u_in){
	return u_in;
}

vector<double> ArmDynamics::wrap_x_out(vector<double> x_out){
	vector<double> x(2*num_links_);
	copy(x_out.begin()+2, x_out.begin()+num_links_+2, x.begin());
    copy(x_out.begin()+num_links_+4, x_out.end(), x.begin()+num_links_);
	return x;
}

void ArmDynamics::setup_constraints(vector<double> const & q, vector<double> const & qd, vector<double> const & theta,
									  vector<double> const & omega, vector<vector<double>> const & vel_com, vector<double> const & tau,
									  MatrixXd& CL, VectorXd& CR){

	ChainDynamics::setup_constraints(q, qd, theta, omega, vel_com, tau, CL, CR);

	// add arm related constraints here
	for(uint i=0; i < num_links_; i++){
		CR(idx_tau_eq(i)) += qd[i] * nu_q_;
	}
	// acceleration of joint-0 must be zero
	CL.block<2,2>(idx_mode_eq(), idx_a(0)) = eye(2);
}