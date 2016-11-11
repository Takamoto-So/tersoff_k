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
   Contributing author:
   Hasan Metin Aktulga, Purdue University - fix qeq/reax implementation
    (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)
   So Takamoto, The University of Tokyo - fix qeq/tersoff_k implementation, which is based on fix qeq/reax

   Please cite the related publication:
     fix qeq/reax:
     H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
     "Parallel Reactive Molecular Dynamics: Numerical Methods and
     Algorithmic Techniques", Parallel Computing, in press.

     fix qeq/tersoff_k:
     S. Takamoto, T. Kumagai, T. Yamasaki, T. Ohno, C. Kaneta, A. Hatano, S. Izumi,
     "Charge-transfer interatomic potential for investigation of the thermal-oxidation growth process of silicon", J. of Appl. Phys.

------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(qeq/tersoff/k,FixQEqTersoff_k)

#else

#ifndef LMP_FIX_QEQ_TERSOFF_K_H
#define LMP_FIX_QEQ_TERSOFF_K_H

#include "fix.h"
#include "pair_tersoff_k.h"

namespace LAMMPS_NS {

class FixQEqTersoff_k : public Fix {
 public:
  FixQEqTersoff_k(class LAMMPS *, int, char **);
  ~FixQEqTersoff_k();
  int setmask();
  void init();
  void init_storage();
  void setup_pre_force(int);
  void pre_force(int);

  void setup_pre_force_respa(int, int);
  void pre_force_respa(int, int, int);

  void min_setup_pre_force(int);
  void min_pre_force(int);

  int matvecs;
  double qeq_time;

 private:
  int nevery;
  int nmax;
  int pack_flag;
  int nlevels_respa;
  class PairTersoff_k *tersoff_k;

  double swa, swb;      // lower/upper Taper cutoff radius (not used)
  double tolerance;     // tolerance for the norm of the rel residual in CG

  double *chi,*eta,*gamma;  // qeq parameters
  std::vector<double> chi_mod, eta_mod;

  // fictitious charges

  std::vector<double> s, t;
  std::vector<double> s_hist, t_hist;
  const int nprev;

  sparse_matrix H;
  std::vector<double> Hdia_inv;
  std::vector<double> b_s, b_t;
  std::vector<double> b_prc, b_prm;

  //CG storage
  std::vector<double> p_vec, q_vec, r_vec, d_vec;

  void pertype_parameters(char*);
  void allocate_storage();
  void deallocate_storage();
  void reallocate_storage();

  void copy_pairdata();

  void init_matvec();
  void calculate_Q();

  int CG(const sparse_matrix&,double*,double*);
  void sparse_matvec(const sparse_matrix&,double*,double*);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  double parallel_norm( double*, int );
  double parallel_dot( double*, double*, int );
  double parallel_vector_acc( double*, int );

  double norm(double*,int);
  void vector_sum(double*,double,double*,double,double*,int);
  void vector_scale(double*,double,double*,int);
  double dot(double*,double*,int);
  void vector_add(double*, double, double*,int);
};

}

#endif
#endif
