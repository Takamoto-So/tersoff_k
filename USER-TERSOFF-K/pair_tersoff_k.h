/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL) - original Tersoff implementation
                        So Takamoto (The University of Tokyo) - Hybrid Tersoff implementation (this file)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(tersoff/k,PairTersoff_k)

#else

#ifndef LMP_PAIR_TERSOFF_K_H
#define LMP_PAIR_TERSOFF_K_H

#include "pair.h"
#include "math.h"
#include <vector>

namespace LAMMPS_NS {

typedef struct{
  std::vector<int> firstnbr;
  std::vector<int> numnbrs;
  std::vector<int> jlist;
  std::vector<double> val; // energy/(qi*qj)
  std::vector<double> fdivqiqj; // force/(qi*qj)
} sparse_matrix;

struct precalc_qeq_ij{
  double prefactor;
  double fpair_divfq;
  double evdwl_divfq;
  double fvion_divqij;
  double eion_divqij;
};

typedef struct{
  std::vector<int> sht_num;        // short-range neighbor list
  std::vector<int> sht_top;
  std::vector<int> ipage;
  std::vector<precalc_qeq_ij> ijpage;
} short_matrix;


class PairTersoff_k : public Pair {
 public:
  PairTersoff_k(class LAMMPS *);
  virtual ~PairTersoff_k();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void compute_pre_qeq(int, int);
  double get_chi_i(int);
  double get_bigj_ii(int);
  const sparse_matrix& get_H() const;

 protected:
  struct Param {
    int ielement,jelement,kelement;
    double el_chi, el_eta;
    double ni_newtral, ni0;

    double biga1, biga2, biga3, bigb1, bigb2, bigb3;
    double lama1, lama2, lama3, lamb1, lamb2, lamb3;
    double Q;
    double powern, powern_del;
    double bigre_ij, bigre_ik;
    double bigd_ij, bigd_ik, bigr_ij, bigr_ik;
    double cutoff_corr_ij, cutoff_r0_ij, cutoff_corr_ik, cutoff_r0_ik;
    double cut,cutsq;
		double cutsq_ij, cutsq_ik;
    double w_alf;
    double coulcut, coulcutsq;
    double gamma, gamma_ecoeff, gamma_fcoeff;

    double powerm;
    double alpha;
    double c,d,h;
    int powermint;
    double ca1,ca4;
    double e_shift, f_shift;
  };
  struct param_single_struct{
    int ielement;
    double el_chi;
    double el_eta;
    double ni_newtral, ni0;
  };
  struct param_double_struct{
    int ielement, jelement;
    double biga1, biga2, biga3, bigb1, bigb2, bigb3;
    double lama1, lama2, lama3, lamb1, lamb2, lamb3;
    double Q;
    double powern, powern_del, powern_rev, powern_rev_del;
    double bigre;
    double bigr, bigd;
    double w_alpha, coulcut;
    double gamma;
  };

  Param *params;                // parameter set for an I-J-K interaction
  char **elements;              // names of unique elements
  int ***elem2param;            // mapping from element triplets to parameters
  int *map;                     // mapping from atom types to elements
  double cutmax;                // max cutoff for all elements
  double cutshort;
  int nmax;
  int nelements;                // # of unique elements
  int nparams;                  // # of stored parameter sets
  int maxparam;                 // max # of parameter sets
  int pgsize, oneatom;

  sparse_matrix H;
	short_matrix Sht;

  struct precalc_qeq_i{
    double chi;
    double bigj_ii;
  };
  std::vector<precalc_qeq_i> pre_qeq_i;

  enum calc_status_enum{
    PRE_QEQ_FINISHED,
    CALC_FINISHED
  } calc_status;
  void allocate();
  virtual void read_file(char *);
  virtual void coulomb_interaction(const Param* const, const double, const double, double&, const int, double&);
  virtual void repulsive(Param* const, const double, double &, const int, double &);
  virtual double zeta(Param* const, const double, const double, double *, double *);
  virtual void force_zeta(Param* const, const double, const double, double &,
                          double &, int, double &);
	virtual double ters_fq(Param*, const double);
	virtual double ters_fq_d(Param*, const double);
	virtual double ters_fq_2d(Param*, const double);
  void attractive(Param* const, const double, const double, const double, double *, double *,
                  double *, double *, double *);

  virtual double ters_fc(const int, const double, Param *);
  virtual double ters_fc_d(const int, const double, Param *);
  virtual double ters_fa(const double, Param *);
  virtual double ters_fa_d(const double, Param *);
  virtual double ters_bij(const double, Param *);
  virtual double ters_bij_d(const double, Param *);

  virtual void ters_zetaterm_d(const double, double *, const double, double *, const double,
                               double *, double *, double *, Param* const);
  void costheta_d(double *, double, double *, double,
                  double *, double *, double *);
  void short_neigh();
  // inlined functions for efficiency

  inline double ters_gijk(const double costheta,
                          const Param * const param) const {
		const double hcth = param->h - costheta;
		return param->c + param->d*(hcth*hcth);
  }

  inline double ters_gijk_d(const double costheta,
                            const Param * const param) const {
    const double hcth = param->h - costheta;
		return -2.0*param->d*hcth;
  }

/* ----------------------------------------------------------------------
repulsive term
 ---------------------------------------------------------------------- */

  inline double k_fQ(const Param* const param, const double rsq){
    const double r = sqrt(rsq);
    return 1.0+param->Q/r;
  }

  inline double k_fQ_d(const Param* const param, const double rsq){
    return -1.0*param->Q/rsq;
  }

/* ---------------------------------------------------------------------- */

  inline double vec3_dot(const double x[3], const double y[3]) const {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }

  inline void vec3_add(const double x[3], const double y[3],
                       double * const z) const {
    z[0] = x[0]+y[0];  z[1] = x[1]+y[1];  z[2] = x[2]+y[2];
  }

  inline void vec3_scale(const double k, const double x[3],
                         double y[3]) const {
    y[0] = k*x[0];  y[1] = k*x[1];  y[2] = k*x[2];
  }

  inline void vec3_scaleadd(const double k, const double x[3],
                            const double y[3], double * const z) const {
    z[0] = k*x[0]+y[0];
    z[1] = k*x[1]+y[1];
    z[2] = k*x[2]+y[2];
  }
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style Tersoff/k requires atom IDs

This is a requirement to use the Tersoff potential.

E: Pair style Tersoff/k requires newton pair on

See the newton command.  This is a restriction to use the Tersoff
potential.

E: Pair style Tersoff/k requires atom attribute q

This is a requirement to use the Tersoff/k potential.
See atom_style command.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open Tersoff potential file %s

The specified potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in Tersoff/k potential file (no single/double/triple word)

Incorrect word in the potential file.

E: Incorrect format in Tersoff/k potential file (different number of parameters)

Incorrect number of words per line in the potential file.

E: Illegal Tersoff parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file for a SW or Tersoff potential has more than
one entry for the same 3 ordered elements.

E: Potential file is missing an entry

The potential file for a SW or Tersoff potential does not have a
needed entry.

*/
