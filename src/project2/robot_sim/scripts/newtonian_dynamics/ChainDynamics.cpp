#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <eigen3/Eigen/Geometry>
#include "ChainDynamics.h"
/* 

x - state

x = [p0, v0, q, qd]  (snake)
x = [q, qd]  (arm)

p0 - postion of head in world frame 
v0 - velocity of head in world frame (this is new)
q  - relative joint angles
qd - joint angular velocities

*/

using namespace Eigen;
using namespace std;

ChainDynamics::ChainDynamics(uint num_links, double link_mass, double link_length, double torques_max,
		double coeff_viscous_joint, double coeff_viscous_link_forward, double coeff_viscous_link_sideways,
		double coeff_dry_link_forward, double coeff_dry_link_sideways, double gravity_y, double gravity_z):
	num_links_(num_links),
	link_mass_(link_mass),
	link_length_(link_length),
    link_inertia_(link_mass*link_length*link_length/12),
	tau_max_(torques_max),
	nu_q_(coeff_viscous_joint),
	nu_x_(coeff_viscous_link_forward),
	nu_y_(coeff_viscous_link_sideways),
	mu_t_(coeff_dry_link_forward),
	mu_n_(coeff_dry_link_sideways),
	g_y_(gravity_y),
	g_z_(gravity_z),
	num_vars_(5*num_links),
	CL_(MatrixXd(num_vars_, num_vars_)),
	CR_(VectorXd(num_vars_)),
	X_(VectorXd(num_vars_)){ }

void ChainDynamics::get_p0(vector<double>const& x, vector<double>& out){
	out.assign(x.begin(),x.begin()+2);
}

void ChainDynamics::get_q(vector<double>const& x, vector<double>& out){
	out.assign(x.begin()+2,x.begin()+num_links_+2);
}

void ChainDynamics::get_v0(vector<double>const& x, vector<double>& out){
	out.assign(x.begin()+num_links_+2,x.begin()+num_links_+4);
}

void ChainDynamics::get_qd(vector<double>const& x, vector<double>& out){
	out.assign(x.begin()+num_links_+4, x.begin()+2*num_links_+4);
}

void ChainDynamics::get_theta(vector<double>const& q, vector<double>& out){
	out.assign(q.begin(),q.end());
	for(uint i=1; i<out.size(); i++){
		out[i] += out[i-1];
	}
}

void ChainDynamics::get_omega(vector<double>const& qd, vector<double>& out){
	out.assign(qd.begin(),qd.end());
	for(uint i=1; i<out.size(); i++){
		out[i] += out[i-1];
	}
}

void ChainDynamics::get_pos_com(vector<double>const& p0, vector<double>const& theta, vector<vector<double>>& out){
    // skip resize if already in correct shape
    if(out.size() != num_links_) out.resize(num_links_);
    for(uint i=0; i<num_links_; i++){
        if(out[i].size() != 2) out[i].resize(num_links_);
    };

    out[0][0] = p0[0] + cos(theta[0])*link_length_/2;
    out[0][1] = p0[1] + sin(theta[0])*link_length_/2;
    for(uint i=1; i<num_links_; i++){
        out[i][0] = out[i-1][0] + (cos(theta[i-1])*link_length_ + cos(theta[i])*link_length_)/2;
        out[i][1] = out[i-1][1] + (sin(theta[i-1])*link_length_ + sin(theta[i])*link_length_)/2;
    }

}

void ChainDynamics::get_vel_com(vector<double>const & v0, vector<double>const & theta, vector<double>const & omega,
					vector<vector<double>>& out){
	// skip resize if already in correct shape
	if(out.size() != num_links_) out.resize(num_links_);
	for(uint i=0; i<num_links_; i++){
		if(out[i].size() != 2) out[i].resize(num_links_);
	};

	out[0][0] = v0[0] - sin(theta[0])*link_length_*omega[0]/2;
	out[0][1] = v0[1] + cos(theta[0])*link_length_*omega[0]/2;
	for(uint i=1; i<num_links_; i++){
		out[i][0] = out[i-1][0] - (sin(theta[i-1])*link_length_*omega[i-1] + sin(theta[i])*link_length_*omega[i])/2;
		out[i][1] = out[i-1][1] + (cos(theta[i-1])*link_length_*omega[i-1] + cos(theta[i])*link_length_*omega[i])/2;
	}
	for(uint i=0; i<num_links_; i++){
		Vector2d v(out[i].data());
		v = rot(-theta[i])*v;
		out[i][0] = v[0];
		out[i][1] = v[1];	
	}
}

void ChainDynamics::limit_tau(vector<double> & tau){
	for(uint i = 0; i < num_links_; i++){
		if(tau[i] > tau_max_) tau[i] = tau_max_;
		if(tau[i] < - tau_max_) tau[i] = - tau_max_;
	}
}


Vector2d xaxis(){
	return Vector2d(1,0);
}

Vector2d yaxis(){
	return Vector2d(0,1);
}

Matrix2d rot(double ang){
	return Rotation2Dd(ang).toRotationMatrix();
}

MatrixXd eye(int X){
	return MatrixXd::Identity(X,X);
}

template<typename T>
T signsmooth(T val, T zero)
{
    return tanh(2*val/zero);
}

template<typename T>
T sign(T val)
{
    if(val > 0) return 1;
    if(val < 0) return -1;
    else return 0;
}
template<typename T>
void print(vector<T> const &v)
{   
    cout << '[';
    for (auto i: v) {
        cout << i << ' ';
    }
    cout << ']';
    cout << endl;
}

void ChainDynamics::setup_constraints(vector<double> const & q, vector<double> const & qd, 	vector<double> const & theta,
	vector<double> const & omega, vector<vector<double>> const & vel_com, vector<double> const & tau,
	MatrixXd& CL, VectorXd& CR){

	// ----------------------------------------------------
	CL.setZero();
	CR.setZero();
	uint i;

	// Force equilibrium constraints ------------------------
	Vector2d vel_com_link; // declare outside the loop
	Vector2d fric;
	double N = fabs(-1 * link_mass_ * g_z_); // normal force per link
	double  zero_vel {1.0e-5}; // this is important, smaller the value stiffer is the system
	for(i=0;i<num_links_;i++){
		CL.block<2,2>(idx_f_eq(i),idx_f(i)) = -1.0 * eye(2);
		CL.block<2,2>(idx_f_eq(i),idx_a(i)) = -1.0 * link_mass_ * eye(2);
		CL(idx_f_eq(i)+1, idx_omdot(i)) = -1.0 * 0.5 * link_length_ * link_mass_;
		if(i < num_links_-1){
			CL.block<2,2>(idx_f_eq(i),idx_f(i+1)) = rot(q[i+1]);	
			}
		CR(idx_f_eq(i)) += -1 * omega[i] * omega[i] * 0.5 * link_length_ * link_mass_;
		CR.segment<2>(idx_f_eq(i)) += g_y_ * link_mass_ * (rot(-theta[i]) * -1 * yaxis());
		// dry friction elliptical
		vel_com_link(0) = vel_com[i][0];
		vel_com_link(1) = vel_com[i][1];

		if(vel_com_link.norm() > 0){
			double alpha = atan2(mu_n_*vel_com_link(1), mu_t_*vel_com_link(0));
			double mu_x = mu_t_ * abs(cos(alpha));
			double mu_y = mu_n_ * abs(sin(alpha));
			fric(0) = -1 * signsmooth(vel_com_link(0), zero_vel) * mu_x * N;
			fric(1) = -1 * signsmooth(vel_com_link(1), zero_vel) * mu_y * N;
			CR.segment<2>(idx_f_eq(i)) += -1 * fric;
		}
		// viscous friction 
		fric(0) = -1 * nu_x_*vel_com_link(0);
		fric(1) = -1 * nu_y_*vel_com_link(1);
		CR.segment<2>(idx_f_eq(i)) += -1 * fric;
	}
	// Torque equilibrium constraints -----------------------
	for(i = 0; i < num_links_; i++){
		CL(idx_tau_eq(i),idx_fy(i)) = 0.5 * link_length_;
		CL(idx_tau_eq(i),idx_omdot(i)) = - link_inertia_;
		if(i < num_links_-1){
			CL.block<1,2>(idx_tau_eq(i), idx_f(i+1)) = 0.5 * link_length_ * rot(q[i+1]).row(1);
		}
		CR(idx_tau_eq(i)) += -tau[i];
		if (i < num_links_-1) {
			CR(idx_tau_eq(i)) += tau[i+1];
		}
		// viscous joint torque considered in derived class 
	}
	
	// Translational acceleration constraints ----------------
	for(i = 1; i < num_links_; i++){
		CL.block<2,2>(idx_a_eq(i), idx_a(i)) = -1 * eye(2);
		CL.block<2,2>(idx_a_eq(i), idx_a(i-1)) = rot(-q[i]);
		CL.block<2,1>(idx_a_eq(i), idx_omdot(i-1)) = link_length_ * rot(-q[i]) * yaxis();
		CR.segment<2>(idx_a_eq(i)) = link_length_ * omega[i-1] * omega[i-1] * rot(-q[i]) * xaxis();
	}
	// ----------------------------------------------------
}

double ChainDynamics::solve_constraints(MatrixXd const & CL, VectorXd const & CR, VectorXd& X){
	X = CL.partialPivLu().solve(CR);
	double  residue;
	residue = (CL*X-CR).norm()/num_vars_;
	if(residue > 1e-4) {
		cout << "residue: " << residue << endl;
		throw;
	}
	return  residue;
}

Vector2d ChainDynamics::get_a0_link(VectorXd const & X){
	return X.segment<2>(idx_a(0));
}

VectorXd ChainDynamics::get_qdd(VectorXd const & X){
	VectorXd qdd(num_links_);
	for(uint i = 0; i<num_links_; i++) qdd(i) = X(idx_omdot(i));
	for(uint i=num_links_-1; i>0; i--) qdd(i) -= qdd(i-1);

	return qdd;
}

vector<double> ChainDynamics::compute_xd(vector<double> x, vector<double> u){
	x = wrap_x_in(x);
	u = wrap_u_in(u);
	vector<double> xd(x.size(), 0.0);

	_compute_xd(x, u, xd);

	xd = wrap_x_out(xd);

	return xd;
}

void ChainDynamics::_compute_xd(vector<double> const & x, vector<double> const & u, vector<double>& xd){

	// get auxiliary state variables from x
	// q, qd, omega, vel_com

	vector<double> v0(2, 0.0), q(num_links_, 0.0), qd(num_links_, 0.0);
	vector<double> theta(num_links_, 0.0), omega(num_links_, 0.0);
	vector<vector<double>> vel_com(num_links_, vector<double>(2, 0.0));

	get_v0(x, v0);
	get_q(x, q);
	get_qd(x, qd);
	get_theta(q, theta);
	get_omega(qd, omega);
	get_vel_com(v0, theta, omega, vel_com);

	// limit the torque applied at each of the joints
	vector<double> tau(num_links_,0.0);
    tau.assign(u.begin(),u.end());
	limit_tau(tau);

	setup_constraints(q, qd, theta, omega, vel_com, tau, CL_, CR_);
	solve_constraints(CL_, CR_, X_);

	// pull derivatives from X and copy to xd
	Vector2d a0 = rot(q[0]) * get_a0_link(X_); // convert to world frame
	VectorXd qdd = get_qdd(X_);
	copy(v0.begin(),v0.end(), xd.begin());
	copy(qd.begin(),qd.end(), xd.begin()+2);
	for(uint i = 0; i<2; i++) xd[2+num_links_+i] = a0(i);
	for(uint i = 0; i<num_links_; i++) xd[num_links_+4+i] = qdd(i);
}

vector<double> ChainDynamics::advance(vector<double> x, vector<double> u, double dt){
	
	x = wrap_x_in(x);
	u = wrap_u_in(u);

	// ----------------------------------------------

	vector<double> xd(x.size());
	// simple euler integration 
	_compute_xd(x, u, xd);

	for(uint i=0; i< x.size(); i++){
		x[i] += xd[i] * dt; 
	}
	// ----------------------------------------------
	x = wrap_x_out(x);
	return x;
}

vector<double> ChainDynamics::compute_energy(std::vector<double> x) {

    x = wrap_x_in(x);

    vector<double> p0(2, 0.0), v0(2, 0.0), q(num_links_, 0.0), qd(num_links_, 0.0);
    vector<double> theta(num_links_, 0.0), omega(num_links_, 0.0);
    vector<vector<double>> pos_com(num_links_, vector<double>(2, 0.0)), vel_com(num_links_, vector<double>(2, 0.0));

    get_p0(x, p0);
    get_v0(x, v0);
    get_q(x, q);
    get_qd(x, qd);
    get_theta(q, theta);
    get_omega(qd, omega);
    get_pos_com(p0, theta, pos_com);
    get_vel_com(v0, theta, omega, vel_com);

    double T{0.0}, V{0.0};

    // kinetic energy

    for(uint i = 0; i<num_links_; i++){
    	T += 0.5 * link_mass_ * (vel_com[i][0]*vel_com[i][0] + vel_com[i][1]*vel_com[i][1]);
    	T += 0.5 * link_inertia_ * omega[i] * omega[i];
    	V += -1 * link_mass_ * g_y_ * pos_com[i][1];
    }

    vector<double> E(2, 0.0);
    E[0] = T;
    E[1] = V;

	return E;
}
