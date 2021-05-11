#include <iostream>
#include <complex>
#include "acb_hypgeom.h"

using namespace std;

#define PREC 10*3.33

const complex<double> II(0,1);

complex<double> WhittW(const double xi, const double Sg, const double z, const slong prec);
complex<double> WhittM(const double xi, const double Sg, const double z, const slong prec);

int main()
{
  double prec = PREC;
  double xi = 5;
  double z = 0.1;
  
  cout << WhittW(xi,2,z,prec) << " " << WhittM(xi,2,z,prec) << endl;
}

complex<double> WhittW(const double xi, const double Sg, const double z, const slong prec) {
  acb_t acba,acbb,acbz,acbU;
  acb_init(acba);
  acb_init(acbb);
  acb_init(acbz);
  acb_init(acbU);
  acb_set_d_d(acba,1+Sg/2,xi);
  acb_set_d(acbb,2+Sg);
  acb_set_d_d(acbz,0,-2*z);
  acb_hypgeom_u(acbU,acba,acbb,acbz,prec);
  complex<double> compU(arf_get_d(arb_midref(acb_realref(acbU)),ARF_RND_DOWN),
			arf_get_d(arb_midref(acb_imagref(acbU)),ARF_RND_DOWN));
  return exp(II*z)*pow(-2.*II*z,1+Sg/2)*compU;
}

complex<double> WhittM(const double xi, const double Sg, const double z, const slong prec) {
  acb_t acba,acbb,acbz,acbM;
  acb_init(acba);
  acb_init(acbb);
  acb_init(acbz);
  acb_init(acbM);
  acb_set_d_d(acba,1+Sg/2,xi);
  acb_set_d(acbb,2+Sg);
  acb_set_d_d(acbz,0,-2*z);
  acb_hypgeom_m(acbM,acba,acbb,acbz,0,prec);
  complex<double> compM(arf_get_d(arb_midref(acb_realref(acbM)),ARF_RND_DOWN),
			arf_get_d(arb_midref(acb_imagref(acbM)),ARF_RND_DOWN));
  return exp(II*z)*pow(-2.*II*z,1+Sg/2)*compM;
}


