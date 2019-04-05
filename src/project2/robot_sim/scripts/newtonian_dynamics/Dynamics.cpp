#include "Dynamics.h"

Dynamics::~Dynamics(){}

std::vector<double> Dynamics::advance(std::vector<double> x, std::vector<double> u, double dt){
	// calls compute_xd() and performs integration with a desired solver
	return x;
}


// http://headmyshoulder.github.io/odeint-v2/examples.html#simple1dexample
// Example: Boost
// #include <iostream>
// #include <boost/array.hpp>

// #include <boost/numeric/odeint.hpp>

// using namespace std;
// using namespace boost::numeric::odeint;

// const double sigma = 10.0;
// const double R = 28.0;
// const double b = 8.0 / 3.0;

// typedef boost::array< double , 3 > state_type;

// void lorenz( const state_type &x , state_type &dxdt , double t )
// {
//     dxdt[0] = sigma * ( x[1] - x[0] );
//     dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
//     dxdt[2] = -b * x[2] + x[0] * x[1];
// }

// void write_lorenz( const state_type &x , const double t )
// {
//     cout << t << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << endl;
// }

// int main(int argc, char **argv)
// {
//     state_type x = { 10.0 , 1.0 , 1.0 }; // initial conditions
//     integrate( lorenz , x , 0.0 , 25.0 , 0.1 , write_lorenz );
// }


// https://stackoverflow.com/questions/16512817/numerical-integration-in-c
// GNU Scientific Library
// #include <stdio.h>
// #include <math.h>
// #include <gsl/gsl_integration.h>

// double f (double x, void * params) {
//   double alpha = *(double *) params;
//   return log(alpha*x) / sqrt(x);
// }

// int
// main (void)
// {
//   double result, error;
//   double expected = -4.0;
//   double alpha = 1.0;
//   gsl_integration_workspace * w 
//     = gsl_integration_workspace_alloc (1000);

//   gsl_function F;
//   F.function = &f;
//   F.params = &alpha;

//   gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000,
//                         w, &result, &error); 

//   printf ("result          = % .18f\n", result);
//   printf ("exact result    = % .18f\n", expected);
//   printf ("estimated error = % .18f\n", error);
//   printf ("actual error    = % .18f\n", result - expected);
//   printf ("intervals =  %d\n", w->size);

//   gsl_integration_workspace_free (w);

//   return 0;
// }