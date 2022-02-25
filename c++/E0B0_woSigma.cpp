#define _USE_MATH_DEFINES

#include <sys/time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include "acb_hypgeom.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#define PREC 100*3.33
#define EE 0.55
#define LNXSTEP 300 //300

const complex<double> II(0,1);

complex<double> WhittW(const complex<double> kappa, const complex<double> mu, const complex<double> z, const slong prec);
complex<double> WhittM(const complex<double> kappa, const complex<double> mu, const complex<double> z, const int reg, const slong prec);
complex<double> Gamma(const complex<double> z, const slong prec);

double xieff(const double xi, const double SgB, const double SgBp, const double cs);
double Sgeff(const double SgE, const double SgEp, const double cs);

complex<double> WS(const double xi, const double Sg, const double x, const slong prec);
complex<double> MS(const double xi, const double Sg, const double x, const slong prec);
complex<double> WSp(const double xi, const double Sg, const double x, const slong prec);
complex<double> MSp(const double xi, const double Sg, const double x, const slong prec);
complex<double> c1(const double xi, const double xieff, const double Sgeff, const double gamma, const slong prec);
complex<double> c2(const double xi, const double xieff, const double Sgeff, const double gamma, const slong prec);

double PBB(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp, const double gamma,
	   const double cs, const double x, const slong prec);
double PEE(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp, const double gamma,
	   const double cs, const double x, const slong prec);
double rhoBcs(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec);
double rhoEcs(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec);
double rhoB(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec);
double rhoE(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec);
double SgEMV(const double ee, const double rhoB, const double rhoE);
double SgEpMV(const double ee, const double rhoB, const double rhoE);
double SgBMV(const double ee, const double rhoB, const double rhoE);
double SgBpMV(const double ee, const double rhoB, const double rhoE);


int main(int argc, char *argv[])
{
  struct timeval tv;
  struct timezone tz;
  double before, after;
  
  gettimeofday(&tv, &tz);
  before = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6; // start stop watch
  

  double prec = PREC;
  double ee = EE;
  int lnxstep = LNXSTEP;
  
  double xii = 1;
  double dxi = 0.5;
  double xiend = 10;
  double SgE = 0;
  double SgEp = 0;
  double SgB = 0;
  double SgBp = 0;

  
#ifdef _OPENMP
  cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << endl;
#endif

  string str = "E0B0_woSigma.dat";
  ofstream ofs(str);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<=(xiend-xii)/dxi; i++) {
    double xi = xii+i*dxi;
    double rhoEout = rhoEcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,0,prec);
    double rhoBout = rhoBcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,0,prec);

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      cout << "xi = " << xi << endl;
      ofs << xi << ' ' << sqrt(rhoEout) << ' ' << sqrt(rhoBout) << endl;
    }
  }

  
  gettimeofday(&tv, &tz);
  after = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
  cout << after - before << " sec." << endl;
}


complex<double> WhittW(const complex<double> kappa, const complex<double> mu, const complex<double> z, const slong prec) {
  acb_t acba,acbb,acbz,acbU;
  acb_init(acba); acb_init(acbb); acb_init(acbz); acb_init(acbU);
  acb_set_d_d(acba, mu.real()-kappa.real()+1./2, mu.imag()-kappa.imag());
  acb_set_d_d(acbb, 1+2*mu.real(), 2*mu.imag());
  acb_set_d_d(acbz, z.real(), z.imag());

  acb_hypgeom_u(acbU,acba,acbb,acbz,prec);
  complex<double> compU(arf_get_d(arb_midref(acb_realref(acbU)),ARF_RND_DOWN),
			arf_get_d(arb_midref(acb_imagref(acbU)),ARF_RND_DOWN));
  
  return exp(-z/2.)*pow(z,mu+1./2)*compU;
}

complex<double> WhittM(const complex<double> kappa, const complex<double> mu, const complex<double> z, const int reg, const slong prec) {
  acb_t acba,acbb,acbz,acbM;
  acb_init(acba); acb_init(acbb); acb_init(acbz); acb_init(acbM);
  acb_set_d_d(acba, mu.real()-kappa.real()+1./2, mu.imag()-kappa.imag());
  acb_set_d_d(acbb, 1+2*mu.real(), 2*mu.imag());
  acb_set_d_d(acbz, z.real(), z.imag());

  acb_hypgeom_m(acbM,acba,acbb,acbz,reg,prec);
  complex<double> compM(arf_get_d(arb_midref(acb_realref(acbM)),ARF_RND_DOWN),
			arf_get_d(arb_midref(acb_imagref(acbM)),ARF_RND_DOWN));
  
  return exp(-z/2.)*pow(z,mu+1./2)*compM;
}

complex<double> Gamma(const complex<double> z, const slong prec) {
  acb_t acbz, acbG;
  acb_init(acbz); acb_init(acbG);
  acb_set_d_d(acbz, z.real(), z.imag());
  acb_gamma(acbG,acbz,prec);
  complex<double> compG(arf_get_d(arb_midref(acb_realref(acbG)),ARF_RND_DOWN),
			arf_get_d(arb_midref(acb_imagref(acbG)),ARF_RND_DOWN));
  return compG;
}


double xieff(const double xi, const double SgB, const double SgBp, const double cs) {
  return xi-1./2*(SgB + SgBp*(1-cs*cs));
}
double Sgeff(const double SgE, const double SgEp, const double cs) {
  return SgE + SgEp*(1-cs*cs);
}


complex<double> WS(const double xi, const double Sg, const double x, const slong prec) {
  return WhittW(-II*xi,(1+Sg)/2,-2.*II*x,prec);
}

complex<double> MS(const double xi, const double Sg, const double x, const slong prec) {
  return WhittM(-II*xi,(1+Sg)/2,-2.*II*x,0,prec);
}

complex<double> WSp(const double xi, const double Sg, const double x, const slong prec) {
  return -(WhittW(1.-II*xi,(1+Sg)/2,-2.*II*x,prec) + II*(x-xi)*WhittW(-II*xi,(1+Sg)/2,-2.*II*x,prec))/x;
}

complex<double> MSp(const double xi, const double Sg, const double x, const slong prec) {
  return ((2.+Sg-2.*II*xi)*WhittM(1.-II*xi,(1+Sg)/2,-2.*II*x,0,prec) - 2.*II*(x-xi)*WhittM(-II*xi,(1+Sg)/2,-2.*II*x,0,prec))/2./x;
}

complex<double> c1(const double xi, const double xieff, const double Sgeff, const double gamma, const slong prec) {
  return II*pow(gamma,-Sgeff/2)*Gamma(1.+II*xieff+Sgeff/2,prec)
    *(WS(xi,0,gamma,prec)*MSp(xieff,Sgeff,gamma,prec) - WSp(xi,0,gamma,prec)*MS(xieff,Sgeff,gamma,prec)
      + Sgeff/2./gamma*WS(xi,0,gamma,prec)*MS(xieff,Sgeff,gamma,prec))
    /Gamma(Sgeff+2,prec)/2.;
}

complex<double> c2(const double xi, const double xieff, const double Sgeff, const double gamma, const slong prec) {
  return -II*pow(gamma,-Sgeff/2)*Gamma(1.+II*xieff+Sgeff/2,prec)
    *(WS(xi,0,gamma,prec)*WSp(xieff,Sgeff,gamma,prec) - WSp(xi,0,gamma,prec)*WS(xieff,Sgeff,gamma,prec)
      + Sgeff/2./gamma*WS(xi,0,gamma,prec)*WS(xieff,Sgeff,gamma,prec))
    /Gamma(Sgeff+2,prec)/2.;
}


double PBB(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,SgB,SgBp,cs);
  double Sgf = Sgeff(SgE,SgEp,cs);
  return exp(M_PI*xi)*pow(x,4+Sgf)*norm(c1(xi,xif,Sgf,gamma,prec)*WS(xif,Sgf,x,prec)
					+ c2(xi,xif,Sgf,gamma,prec)*MS(xif,Sgf,x,prec))/4/M_PI/M_PI;
}

double PEE(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,SgB,SgBp,cs);
  double Sgf = Sgeff(SgE,SgEp,cs);
  complex<double> cc1 = c1(xi,xif,Sgf,gamma,prec);
  complex<double> cc2 = c2(xi,xif,Sgf,gamma,prec);
  
  return exp(M_PI*xi)*pow(x,4+Sgf)
    *norm(cc1*WSp(xif,Sgf,x,prec) + cc2*MSp(xif,Sgf,x,prec)
	  + Sgf/2./x *(cc1*WS(xif,Sgf,x,prec) + cc2*MS(xif,Sgf,x,prec)) )/4/M_PI/M_PI;
}

double rhoBcs(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec) {
  double rhoBcs = 0;

  double xif = xieff(xi,SgB,SgBp,cs);
  double gamma = 2*xi; //2*abs(xif);
  double xmax = gamma; //gamma/2;
  double xmin = xmax*(1e-3);

  double dlnx = (log(xmax)-log(xmin))/lnxstep;
  double xx;
  
  rhoBcs += (PBB(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xmin,prec) + PBB(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xmax,prec))/2*dlnx;

  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoBcs += PBB(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xx,prec)*dlnx;
  }

  return rhoBcs;
}

double rhoB(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec) {
  double rhoB = 0;

  double dcs = 1./csstep; //2./csstep;
  double csmin = -1, csmax = 0; //1;
  double cs;
  
  rhoB += (rhoBcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
		  csmin,prec) + rhoBcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
				       csmax,prec))/2*dcs;
  int done = 50;
  cout << "\r" << setw(3) << done/csstep << "%" << flush;

#ifdef _OPENMP
#pragma omp parallel for private(cs)
#endif
  for (int i=1; i<csstep; i++) {
    cs = csmin+i*dcs;
    
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      done += 50;
      cout << "\r" << setw(3) << done/csstep << "%" << flush;
    }

#ifdef _OPENMP
#pragma omp atomic
#endif
    rhoB += rhoBcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
		   cs,prec)*dcs;
  }

  return 2*rhoB/4;
}

double rhoEcs(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec) {
  double rhoEcs = 0;

  double xif = xieff(xi,SgB,SgBp,cs);
  double gamma = 2*xi; //2*abs(xif);
  double xmax = gamma; //gamma/2;
  double xmin = xmax*(1e-3);

  double dlnx = (log(xmax)-log(xmin))/lnxstep;
  double xx;
  
  rhoEcs += (PEE(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xmin,prec) + PEE(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xmax,prec))/2*dlnx;
  
  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoEcs += PEE(xi,SgE,SgEp,SgB,SgBp,gamma,cs,xx,prec)*dlnx;
  }

  return rhoEcs;
}

double rhoE(const double xi, const double SgE, const double SgEp, const double SgB, const double SgBp,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec) {
  double rhoE = 0;

  double dcs = 1./csstep; //2./csstep;
  double csmin = -1, csmax = 0; //1;
  double cs;
  
  rhoE += (rhoEcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
		  csmin,prec) + rhoEcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
				       csmax,prec))/2*dcs;
  int done = 50*(csstep+1);
  cout << "\r" << setw(3) << done/csstep << "%" << flush;

#ifdef _OPENMP
#pragma omp parallel for private(cs)
#endif
  for (int i=1; i<csstep; i++) {
    cs = csmin+i*dcs;
    
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      done += 50;
      cout << "\r" << setw(3) << done/csstep << "%" << flush;
    }

#ifdef _OPENMP
#pragma omp atomic
#endif
    rhoE += rhoEcs(xi,SgE,SgEp,SgB,SgBp,lnxstep,//xmin,
		   cs,prec)*dcs;
  }
  cout << "   " << flush;

  return 2*rhoE/4;
}

double SgEMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/6/pow(M_PI,2)*sqrt(2*rhoB)*rhoE/(rhoE+rhoB)/tanh(M_PI*sqrt(rhoB/rhoE));
}

double SgEpMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/12/pow(M_PI,2)*sqrt(2*rhoB)*(rhoB/(rhoE+rhoB)/tanh(M_PI*sqrt(rhoB/rhoE))
						+ M_PI*sqrt(rhoB/rhoE)/sinh(M_PI*sqrt(rhoB/rhoE))/sinh(M_PI*sqrt(rhoB/rhoE)));
}

double SgBMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/6/pow(M_PI,2)*sqrt(2*rhoE)*rhoB/(rhoE+rhoB)/tanh(M_PI*sqrt(rhoB/rhoE));
}

double SgBpMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/12/pow(M_PI,2)*sqrt(2*rhoE)*(rhoE/(rhoE+rhoB)/tanh(M_PI*sqrt(rhoB/rhoE))
						- M_PI*sqrt(rhoB/rhoE)/sinh(M_PI*sqrt(rhoB/rhoE))/sinh(M_PI*sqrt(rhoB/rhoE)));
}
