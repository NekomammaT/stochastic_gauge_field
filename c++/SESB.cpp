#define _USE_MATH_DEFINES

#include <sys/time.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <complex>
#include "acb_hypgeom.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#define PREC 15*3.33
#define EE 0.55
#define TOL 0.1
#define SGSAFE 0.5
#define CSSTEP 200
#define LNXSTEP 300

const complex<double> II(0,1);

complex<double> WhittW(const complex<double> kappa, const complex<double> mu, const complex<double> z, const slong prec);
complex<double> WhittM(const complex<double> kappa, const complex<double> mu, const complex<double> z, const int reg, const slong prec);
complex<double> Gamma(const complex<double> z, const slong prec);

double xieff(const double xi, const double SgB, const double cs);

complex<double> WS(const double xi, const double SgE, const double x, const slong prec);
complex<double> MS(const double xi, const double SgE, const double x, const slong prec);
complex<double> WSp(const double xi, const double SgE, const double x, const slong prec);
complex<double> MSp(const double xi, const double SgE, const double x, const slong prec);
complex<double> c1(const double xi, const double xieff, const double SgE, const double gamma, const slong prec);
complex<double> c2(const double xi, const double xieff, const double SgE, const double gamma, const slong prec);

double PBB(const double xi, const double SgE, const double SgB, const double gamma,
	   const double cs, const double x, const slong prec);
double PEE(const double xi, const double SgE, const double SgB, const double gamma,
	   const double cs, const double x, const slong prec);
double rhoBcs(const double xi, const double SgE, const double SgB,
	      const int lnxstep, const double xmin, const double cs, const slong prec);
double rhoEcs(const double xi, const double SgE, const double SgB,
	      const int lnxstep, const double xmin, const double cs, const slong prec);
double rhoB(const double xi, const double SgE, const double SgB,
	    const int csstep, const int lnxstep, const double xmin, const slong prec);
double rhoE(const double xi, const double SgE, const double SgB,
	    const int csstep, const int lnxstep, const double xmin, const slong prec);
double SgEMV(const double ee, const double rhoB, const double rhoE);
double SgBMV(const double ee, const double rhoB, const double rhoE);


int main(int argc, char *argv[])
{
  if (argc != 2) {
    cout << "xi を正しく指定してください" << endl;
    return 1;
  }
  
  struct timeval tv;
  struct timezone tz;
  double before, after;
  
  gettimeofday(&tv, &tz);
  before = (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6; // start stop watch
  
  
  double prec = PREC;
  double ee = EE;
  int csstep = CSSTEP;
  int lnxstep = LNXSTEP;
  double xmin = 0.01;
  
  double xi = atof(argv[1]);
  double SgEmax = 10*xi;
  double SgBmax = 2*xi;
  double SgEin = 1;
  double SgBin = 1;
  

#ifdef _OPENMP
  cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << endl;
#endif

  cout << "xi = " << xi << endl;

  double rhoBout, rhoEout, SgEout, SgBout;

  while (true) {
    rhoBout = rhoB(xi,SgEin,SgBin,csstep,lnxstep,xmin,prec);
    rhoEout = rhoE(xi,SgEin,SgBin,csstep,lnxstep,xmin,prec);
    SgEout = SgEMV(ee,rhoBout,rhoEout);
    SgBout = SgBMV(ee,rhoBout,rhoEout);

    cout << "(SgEin, SgBin) = (" << SgEin << ", " << SgBin << "), (rhoEout, rhoBout) = (" << rhoEout << ", " << rhoBout
	 << "), (SgEout, SgBout) = (" << SgEout << ", " << SgBout << ")" << flush;

    if (SgEout > SgEmax || SgBout > SgBmax) {
      SgEout = min(SgEout,SgEmax);
      SgBout = min(SgBout,SgBmax);

      cout << ", Sg is reduced to (SgEout, SgBout) = (" << SgEout << ", " << SgBout << ")" << flush;
    }
    if (!(SgEout > 0) || !(SgBout > 0)) {
      SgEout = SgEin*SGSAFE;
      SgBout = SgBin*SGSAFE;

      cout << ", Sg is reduced to (SgEout, SgBout) = (" << SgEout << ", " << SgBout << ")" << flush;
    }
    cout << endl;

    if (abs(SgEin-SgEout)/SgEout < TOL && abs(SgBin-SgBout)/SgBout < TOL) {
      break;
    }

    SgEin = pow(SgEin*SgEin*SgEout,1./3);
    SgBin = pow(SgBin*SgBin*SgBout,1./3);
  }
  cout << "(SgEout, SgBout) = (" << SgEout << ", " << SgBout << ")" << endl;
  


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


double xieff(const double xi, const double SgB, const double cs) {
  return xi-1./2*SgB*(1-cs*cs);
}


complex<double> WS(const double xi, const double SgE, const double x, const slong prec) {
  return WhittW(-II*xi,(1+SgE)/2,-2.*II*x,prec);
}

complex<double> MS(const double xi, const double SgE, const double x, const slong prec) {
  return WhittM(-II*xi,(1+SgE)/2,-2.*II*x,0,prec);
}

complex<double> WSp(const double xi, const double SgE, const double x, const slong prec) {
  return -(WhittW(1.-II*xi,(1+SgE)/2,-2.*II*x,prec) + II*(x-xi)*WhittW(-II*xi,(1+SgE)/2,-2.*II*x,prec))/x;
}

complex<double> MSp(const double xi, const double SgE, const double x, const slong prec) {
  return ((2.+SgE-2.*II*xi)*WhittM(1.-II*xi,(1+SgE)/2,-2.*II*x,0,prec) - 2.*II*(x-xi)*WhittM(-II*xi,(1+SgE)/2,-2.*II*x,0,prec))/2./x;
}

complex<double> c1(const double xi, const double xieff, const double SgE, const double gamma, const slong prec) {
  return pow(gamma,-SgE/2)*Gamma(1.+II*xieff+SgE/2,prec)
    *(WS(xi,0,gamma,prec)*MSp(xieff,SgE,gamma,prec) - WSp(xi,0,gamma,prec)*MS(xieff,SgE,gamma,prec)
      + SgE/2./gamma*WS(xi,0,gamma,prec)*MS(xieff,SgE,gamma,prec))
    /Gamma(SgE+2,prec);
}

complex<double> c2(const double xi, const double xieff, const double SgE, const double gamma, const slong prec) {
  return -pow(gamma,-SgE/2)*Gamma(1.+II*xieff+SgE/2,prec)
    *(WS(xi,0,gamma,prec)*WSp(xieff,SgE,gamma,prec) - WSp(xi,0,gamma,prec)*WS(xieff,SgE,gamma,prec)
      + SgE/2./gamma*WS(xi,0,gamma,prec)*WS(xieff,SgE,gamma,prec))
    /Gamma(SgE+2,prec);
}


double PBB(const double xi, const double SgE, const double SgB, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,SgB,cs);
  return exp(M_PI*xi)*pow(x,4+SgE)*norm(c1(xi,xif,SgE,gamma,prec)*WS(xif,SgE,x,prec)
					+ c2(xi,xif,SgE,gamma,prec)*MS(xif,SgE,x,prec));
}

double PEE(const double xi, const double SgE, const double SgB, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,SgB,cs);
  complex<double> cc1 = c1(xi,xif,SgE,gamma,prec);
  complex<double> cc2 = c2(xi,xif,SgE,gamma,prec);
  
  return exp(M_PI*xi)*pow(x,4+SgE)
    *norm(cc1*WSp(xif,SgE,x,prec) + cc2*MSp(xif,SgE,x,prec)
	  + SgE/2./x *(cc1*WS(xif,SgE,x,prec) + cc2*MS(xif,SgE,x,prec)) );
}

double rhoBcs(const double xi, const double SgE, const double SgB,
	      const int lnxstep, const double xmin, const double cs, const slong prec) {
  double rhoBcs = 0;

  double xif = xieff(xi,SgB,cs);
  double gamma = 2*xif;

  double dlnx = (log(gamma)-log(xmin))/lnxstep;
  double xx;
  
  rhoBcs += (PBB(xi,SgE,SgB,gamma,cs,xmin,prec) + PBB(xi,SgE,SgB,gamma,cs,gamma,prec))/2*dlnx;
  
  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoBcs += PBB(xi,SgE,SgB,gamma,cs,xx,prec)*dlnx;
  }

  return rhoBcs;
}

double rhoB(const double xi, const double SgE, const double SgB,
	    const int csstep, const int lnxstep, const double xmin, const slong prec) {
  double rhoB = 0;

  double dcs = 2./csstep;
  double csmin = -1, csmax = 1;
  double cs;
  
  rhoB += (rhoBcs(xi,SgE,SgB,lnxstep,xmin,csmin,prec) + rhoBcs(xi,SgE,SgB,lnxstep,xmin,csmax,prec))/2*dcs;
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

    /*
    if (!(rhoB > 0)) {
      rhoB = 0;
      continue;
      } else {*/
#ifdef _OPENMP
#pragma omp atomic
#endif
      rhoB += rhoBcs(xi,SgE,SgB,lnxstep,xmin,cs,prec)*dcs;
      //}
  }

  return rhoB/4;
}

double rhoEcs(const double xi, const double SgE, const double SgB,
	      const int lnxstep, const double xmin, const double cs, const slong prec) {
  double rhoEcs = 0;

  double xif = xieff(xi,SgB,cs);
  double gamma = 2*xif;

  double dlnx = (log(gamma)-log(xmin))/lnxstep;
  double xx;
  
  rhoEcs += (PEE(xi,SgE,SgB,gamma,cs,xmin,prec) + PEE(xi,SgE,SgB,gamma,cs,gamma,prec))/2*dlnx;
  
  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoEcs += PEE(xi,SgE,SgB,gamma,cs,xx,prec)*dlnx;
  }

  return rhoEcs;
}

double rhoE(const double xi, const double SgE, const double SgB,
	    const int csstep, const int lnxstep, const double xmin, const slong prec) {
  double rhoE = 0;

  double dcs = 2./csstep;
  double csmin = -1, csmax = 1;
  double cs;
  
  rhoE += (rhoEcs(xi,SgE,SgB,lnxstep,xmin,csmin,prec) + rhoEcs(xi,SgE,SgB,lnxstep,xmin,csmax,prec))/2*dcs;
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

    /*
    if (!(rhoE > 0)) {
      rhoE = 0;
      continue;
      } else {*/
#ifdef _OPENMP
#pragma omp atomic
#endif
      rhoE += rhoEcs(xi,SgE,SgB,lnxstep,xmin,cs,prec)*dcs;
      //}
  }
  cout << "   " << flush;

  return rhoE/4;
}

double SgEMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/12/pow(M_PI,3)*sqrt(2*rhoB)/tanh(M_PI*sqrt(rhoB/rhoE));
}

double SgBMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/24/pow(M_PI,3)*sqrt(2*rhoE)/tanh(M_PI*sqrt(rhoB/rhoE));
}

