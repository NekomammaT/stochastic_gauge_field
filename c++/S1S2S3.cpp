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

#define PREC 50*3.33
#define EE 0.55
#define TOL 0.1
#define SGSAFE 0.5
#define CSSTEP 20 //200
#define LNXSTEP 30 //300

const complex<double> II(0,1);

complex<double> WhittW(const complex<double> kappa, const complex<double> mu, const complex<double> z, const slong prec);
complex<double> WhittM(const complex<double> kappa, const complex<double> mu, const complex<double> z, const int reg, const slong prec);
complex<double> Gamma(const complex<double> z, const slong prec);

double xieff(const double xi, const double Sg3, const double cs);
double Sgeff(const double Sg1, const double Sg2, const double cs);

complex<double> WS(const double xi, const double Sg, const double x, const slong prec);
complex<double> MS(const double xi, const double Sg, const double x, const slong prec);
complex<double> WSp(const double xi, const double Sg, const double x, const slong prec);
complex<double> MSp(const double xi, const double Sg, const double x, const slong prec);
complex<double> c1(const double xi, const double xieff, const double Sg, const double gamma, const slong prec);
complex<double> c2(const double xi, const double xieff, const double Sg, const double gamma, const slong prec);

double PBB(const double xi, const double Sg1, const double Sg2, const double Sg3, const double gamma,
	   const double cs, const double x, const slong prec);
double PEE(const double xi, const double Sg1, const double Sg2, const double Sg3, const double gamma,
	   const double cs, const double x, const slong prec);
double rhoBcs(const double xi, const double Sg1, const double Sg2, const double Sg3,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec);
double rhoEcs(const double xi, const double Sg1, const double Sg2, const double Sg3,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec);
double rhoB(const double xi, const double Sg1, const double Sg2, const double Sg3,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec);
double rhoE(const double xi, const double Sg1, const double Sg2, const double Sg3,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec);
double Sg1MV(const double ee, const double rhoB, const double rhoE);
double Sg2MV(const double ee, const double rhoB, const double rhoE);
double Sg3MV(const double ee, const double rhoB, const double rhoE);


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
  //double xmin = 0.01;
  
  double xi = atof(argv[1]);
  double Sg1max = 10*xi;
  double Sg2max = 10*xi;
  double Sg3max = 2*xi;
  double Sg1in = 1;
  double Sg2in = 1;
  double Sg3in = 1;
  

#ifdef _OPENMP
  cout << "OpenMP : Enabled (Max # of threads = " << omp_get_max_threads() << ")" << endl;
#endif

  cout << "xi = " << xi << endl;

  double rhoBout, rhoEout, Sg1out, Sg2out, Sg3out;

  while (true) {
    rhoBout = rhoB(xi,Sg1in,Sg2in,Sg3in,csstep,lnxstep,//xmin,
		   prec);
    rhoEout = rhoE(xi,Sg1in,Sg2in,Sg3in,csstep,lnxstep,//xmin,
		   prec);
    Sg1out = Sg1MV(ee,rhoBout,rhoEout);
    Sg2out = Sg2MV(ee,rhoBout,rhoEout);
    Sg3out = Sg3MV(ee,rhoBout,rhoEout);

    cout << "(Sg1in, Sg2in, Sg3in) = (" << Sg1in << ", " << Sg2in << ", " << Sg3in << "), (rhoEout, rhoBout) = (" << rhoEout << ", " << rhoBout
	 << "), (Sg1out, Sg2out, Sg3out) = (" << Sg1out << ", " << Sg2out << ", " << Sg3out << ")" << flush;

    if (Sg1out > Sg1max || Sg2out > Sg2max || Sg3out > Sg3max) {
      Sg1out = min(Sg1out,Sg1max);
      Sg2out = min(Sg2out,Sg2max);
      Sg3out = min(Sg3out,Sg3max);

      cout << ", Sg is reduced to (Sg1out, Sg2out, Sg3out) = (" << Sg1out << ", " << Sg2out << ", " << Sg3out << ")" << flush;
    }
    if (!(Sg1out > 0) || !(Sg2out > 0) || !(Sg3out > 0)) {
      Sg1out = Sg1in*SGSAFE;
      Sg2out = Sg2in*SGSAFE;
      Sg3out = Sg3in*SGSAFE;

      cout << ", Sg is reduced to (Sg1out, Sg2out, Sg3out) = (" << Sg1out << ", " << Sg2out << ", " << Sg3out << ")" << flush;
    }
    cout << endl;

    if (abs(Sg1in-Sg1out)/Sg1out < TOL && abs(Sg2in-Sg2out)/Sg2out < TOL && abs(Sg3in-Sg3out)/Sg3out < TOL) {
      break;
    }

    Sg1in = pow(Sg1in*Sg1in*Sg1in*Sg1out,1./4);
    Sg2in = pow(Sg2in*Sg2in*Sg2in*Sg2out,1./4);
    Sg3in = pow(Sg3in*Sg3in*Sg3in*Sg3out,1./4);
  }
  cout << "(Sg1out, Sg2out, Sg3out) = (" << Sg1out << ", " << Sg2out << ", " << Sg3out << ")" << endl;
  


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


double xieff(const double xi, const double Sg3, const double cs) {
  return xi-1./2*Sg3*(1-cs*cs);
}
double Sgeff(const double Sg1, const double Sg2, const double cs) {
  return Sg1+Sg2*(1-cs*cs);
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

complex<double> c1(const double xi, const double xieff, const double Sg, const double gamma, const slong prec) {
  return pow(gamma,-Sg/2)*Gamma(1.+II*xieff+Sg/2,prec)
    *(WS(xi,0,gamma,prec)*MSp(xieff,Sg,gamma,prec) - WSp(xi,0,gamma,prec)*MS(xieff,Sg,gamma,prec)
      + Sg/2./gamma*WS(xi,0,gamma,prec)*MS(xieff,Sg,gamma,prec))
    /Gamma(Sg+2,prec);
}

complex<double> c2(const double xi, const double xieff, const double Sg, const double gamma, const slong prec) {
  return -pow(gamma,-Sg/2)*Gamma(1.+II*xieff+Sg/2,prec)
    *(WS(xi,0,gamma,prec)*WSp(xieff,Sg,gamma,prec) - WSp(xi,0,gamma,prec)*WS(xieff,Sg,gamma,prec)
      + Sg/2./gamma*WS(xi,0,gamma,prec)*WS(xieff,Sg,gamma,prec))
    /Gamma(Sg+2,prec);
}


double PBB(const double xi, const double Sg1, const double Sg2, const double Sg3, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,Sg3,cs);
  double Sgf = Sgeff(Sg1,Sg2,cs);
  return exp(M_PI*xi)*pow(x,4+Sgf)*norm(c1(xi,xif,Sgf,gamma,prec)*WS(xif,Sgf,x,prec)
					+ c2(xi,xif,Sgf,gamma,prec)*MS(xif,Sgf,x,prec));
}

double PEE(const double xi, const double Sg1, const double Sg2, const double Sg3, const double gamma,
	   const double cs, const double x, const slong prec) {
  double xif = xieff(xi,Sg3,cs);
  double Sgf = Sgeff(Sg1,Sg2,cst;)
  complex<double> cc1 = c1(xi,xif,Sgf,gamma,prec);
  complex<double> cc2 = c2(xi,xif,Sgf,gamma,prec);
  
  return exp(M_PI*xi)*pow(x,4+Sgf)
    *norm(cc1*WSp(xif,Sgf,x,prec) + cc2*MSp(xif,Sgf,x,prec)
	  + SgE/2./x *(cc1*WS(xif,Sgf,x,prec) + cc2*MS(xif,Sgf,x,prec)) );
}

double rhoBcs(const double xi, const double Sg1, const double Sg2, const double Sg3,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec) {
  double rhoBcs = 0;

  double xif = xieff(xi,Sg3,cs);
  double Sgf = Sgeff(Sg1,Sg2,cs);
  double gamma = 2*xif;
  double xmax = gamma/2;
  double xmin = xmax*(1e-3);

  double dlnx = (log(xmax)-log(xmin))/lnxstep;
  double xx;
  
  rhoBcs += (PBB(xi,Sg1,Sg2,Sg3,gamma,cs,xmin,prec) + PBB(xi,Sg1,Sg2,Sg3,gamma,cs,xmax,prec))/2*dlnx;
  
  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoBcs += PBB(xi,Sg1,Sg2,Sg3,gamma,cs,xx,prec)*dlnx;
  }

  return rhoBcs;
}

double rhoB(const double xi, const double Sg1, const double Sg2, const double Sg3,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec) {
  double rhoB = 0;

  double dcs = 2./csstep;
  double csmin = -1, csmax = 1;
  double cs;
  
  rhoB += (rhoBcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
		  csmin,prec) + rhoBcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
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
    rhoB += rhoBcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
		   cs,prec)*dcs;
  }

  return rhoB/4;
}

double rhoEcs(const double xi, const double Sg1, const double Sg2, const double Sg3,
	      const int lnxstep, //const double xmin,
	      const double cs, const slong prec) {
  double rhoEcs = 0;

  double xif = xieff(xi,Sg3,cs);
  double Sgf = Sgeff(Sg1,Sg2,cs);
  double gamma = 2*xif;
  double xmax = gamma/2;
  double xmin = xmax*(1e-3);

  double dlnx = (log(xmax)-log(xmin))/lnxstep;
  double xx;
  
  rhoEcs += (PEE(xi,Sg1,Sg2,Sg3,gamma,cs,xmin,prec) + PEE(xi,Sg1,Sg2,Sg3,gamma,cs,xmax,prec))/2*dlnx;
  
  for (int i=1; i<lnxstep; i++) {
    xx = xmin*exp(i*dlnx);
    rhoEcs += PEE(xi,Sg1,Sg2,Sg3,gamma,cs,xx,prec)*dlnx;
  }

  return rhoEcs;
}

double rhoE(const double xi, const double Sg1, const double Sg2, const double Sg3,
	    const int csstep, const int lnxstep, //const double xmin,
	    const slong prec) {
  double rhoE = 0;

  double dcs = 2./csstep;
  double csmin = -1, csmax = 1;
  double cs;
  
  rhoE += (rhoEcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
		  csmin,prec) + rhoEcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
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
    rhoE += rhoEcs(xi,Sg1,Sg2,Sg3,lnxstep,//xmin,
		   cs,prec)*dcs;
  }
  cout << "   " << flush;

  return rhoE/4;
}

double Sg1MV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/6/pow(M_PI,3)*sqrt(2*rhoB)/tanh(M_PI*sqrt(rhoB/rhoE));
}

double SgBMV(const double ee, const double rhoB, const double rhoE) {
  return pow(ee,3)/24/pow(M_PI,3)*sqrt(2*rhoE)/tanh(M_PI*sqrt(rhoB/rhoE));
}

