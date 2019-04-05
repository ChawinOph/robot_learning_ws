#ifndef CHAIN_DYNAMICS_H
#define CHAIN_DYNAMICS_H

#include <vector>
#include <eigen3/Eigen/Dense>

#include "Dynamics.h"

class ChainDynamics : public Dynamics
{


public:
    ChainDynamics(uint num_links, double link_mass, double link_length, double torques_max, double coeff_viscous_joint,
            double coeff_viscous_link_forward, double coeff_viscous_link_sideways, double coeff_dry_link_forward,
            double coeff_dry_link_sideways, double gravity_y, double gravity_z);

    std::vector<double> compute_xd(std::vector<double> x, std::vector<double> u);
    std::vector<double> advance(std::vector<double> x, std::vector<double> u, double dt);
    std::vector<double> compute_energy(std::vector<double> x);

protected:
    uint num_links_; // number of links
    double link_mass_; // mass of link
    double link_length_; // length of link
    double link_inertia_; // moment of inertial of link
    double tau_max_; // maximum torque available at a joint
    double nu_q_;  // viscous friction coeff for joint
    double nu_x_; // viscous friction coeff for link (forward)
    double nu_y_; // link viscous friction coeff for link (sideways)
    double mu_t_; // dry friction coeff for link (forward)
    double mu_n_; // dry friction coeff for link (sideways)
    double g_y_; // gravity along y-axis
    double g_z_; // gravity along z-axis
    int num_vars_;
    Eigen::MatrixXd CL_; // constraint matrix left hand side
    Eigen::VectorXd CR_; // constraint matrix right hand side
    Eigen::VectorXd X_; // solution will be stored here after solving


    virtual std::vector<double> wrap_x_in(std::vector<double> x) = 0; // wrappers handle shape conversions
    virtual std::vector<double> wrap_u_in(std::vector<double> u) = 0;
    virtual std::vector<double> wrap_x_out(std::vector<double> x) = 0;
	void get_p0(std::vector<double>const & x, std::vector<double>& out); // kinematics from x
    void get_v0(std::vector<double>const & x, std::vector<double>& out);
    void get_q(std::vector<double>const & x, std::vector<double>& out);
    void get_qd(std::vector<double>const & x, std::vector<double>& out);
	void get_theta(std::vector<double>const & q, std::vector<double>& out);
	void get_omega(std::vector<double>const & qd, std::vector<double>& out);
	void get_pos_com(std::vector<double>const & p0, std::vector<double>const & theta,
	        std::vector<std::vector<double>>& out);
	void get_vel_com(std::vector<double>const & v0, std::vector<double>const & theta, std::vector<double>const & omega,
	        std::vector<std::vector<double>>& out);
    void limit_tau(std::vector<double> & tau);
    
    // populate CL and CR which represent the system to be solved
    virtual void setup_constraints(std::vector<double> const & q, std::vector<double> const & qd,
            std::vector<double> const & theta, std::vector<double> const & omega,
            std::vector<std::vector<double>> const & vel_com, std::vector<double> const & tau,
            Eigen::MatrixXd& CL, Eigen::VectorXd& CR);
    double solve_constraints(Eigen::MatrixXd const & CL, Eigen::VectorXd const & CR, Eigen::VectorXd& X);
    Eigen::Vector2d get_a0_link(Eigen::VectorXd const & X);
    Eigen::VectorXd get_qdd(Eigen::VectorXd const & X);
    void _compute_xd(std::vector<double> const & x, std::vector<double> const & u, std::vector<double> & xd);

    inline uint idx_f(uint i) {return 2*i;}
    inline uint idx_fx(uint i) {return 2*i;}
    inline uint idx_fy(uint i) {return 2*i+1;}
    inline uint idx_a(uint i) {return 2*num_links_+2*i;}
    inline uint idx_omdot(uint i){return 4*num_links_+i;}
    inline uint idx_f_eq(uint i){return 2*i;}
    inline uint idx_tau_eq(uint i){return 2*num_links_ + i;}
    inline uint idx_a_eq(uint i){return 3*num_links_+2*(i-1);} // i > 0
    inline uint idx_mode_eq(){return 5*num_links_-2;}
    virtual uint get_state_dim() = 0;
    virtual uint get_action_dim() = 0;

};

Eigen::Vector2d xaxis();
Eigen::Vector2d yaxis();
Eigen::Matrix2d rot(double ang);
Eigen::MatrixXd eye(int X);
template<typename T>
T sign(T val, T zero);
template<typename T>
T sign(T val);
template<typename T>
void print(std::vector<T> const &v);

#endif