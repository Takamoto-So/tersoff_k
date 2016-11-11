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
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "fix_qeq_tersoff_k.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "respa.h"
#include "memory.h"
#include "citeme.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_qeq_tersoff_k[] =
  "fix qeq/reax command (the implementation of fix qeq/tersoff_k command is based on fix qeq/reax):\n\n"
  "@Article{Aktulga12,\n"
  " author = {H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama},\n"
  " title = {Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques},\n"
  " journal = {Parallel Computing},\n"
  " year =    2012,\n"
  " volume =  38,\n"
  " pages =   {245--259}\n"
  "}\n\n"
  "fix qeq/tersoff_k command:\n\n"
  "@Article{:/content/aip/journal/jap/120/16/10.1063/1.4965863,\n"
  " author = {Takamoto, So and Kumagai, Tomohisa and Yamasaki, Takahiro and Ohno, Takahisa and Kaneta, Chioko and Hatano, Asuka and Izumi, Satoshi},\n"
  " title = {Charge-transfer interatomic potential for investigation of the thermal-oxidation growth process of silicon},\n"
  " journal = {Journal of Applied Physics},\n"
  " year =    2016,\n"
  " volume =  120,\n"
  " number =  16,\n"
  " pages  =  {165109},\n"
  " doi    =  {http://dx.doi.org/10.1063/1.4965863}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

FixQEqTersoff_k::FixQEqTersoff_k(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), nprev(5)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_qeq_tersoff_k);

  if (narg != 8) error->all(FLERR,"Illegal fix qeq/tersoff_k command");

  nevery = force->inumeric(FLERR,arg[3]);
  swa = force->numeric(FLERR,arg[4]);
  swb = force->numeric(FLERR,arg[5]);
  tolerance = force->numeric(FLERR,arg[6]);
  pertype_parameters(arg[7]);

  nmax = 0;
  pack_flag = 0;

  comm_forward = comm_reverse = 1;

  s_hist.resize(atom->nmax*nprev, 0.0);
  t_hist.resize(atom->nmax*nprev, 0.0);
  atom->add_callback(0);
  tersoff_k = NULL;
  tersoff_k = (PairTersoff_k *) force->pair_match("tersoff/k",1);

}

/* ---------------------------------------------------------------------- */

FixQEqTersoff_k::~FixQEqTersoff_k()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  deallocate_storage();
}

/* ---------------------------------------------------------------------- */

int FixQEqTersoff_k::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::pertype_parameters(char *arg)
{
  if(strcmp(arg,"tersoff/k") == 0){
    Pair *pair = force->pair_match("tersoff/k",1);
    if(pair == NULL) error->all(FLERR,"No pair tersoff/k for fix qeq/tersoff_k");
    return;
  }else{
    error->all(FLERR,"Invalid arguments for fix qeq/tersoff_k");
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::allocate_storage()
{
  nmax = atom->nmax+1;

  s.resize(nmax, 0.0);
  t.resize(nmax, 0.0);

  s_hist.resize(nmax*nprev, 0.0);
  t_hist.resize(nmax*nprev, 0.0);

  Hdia_inv.resize(nmax, 0.0);
  b_s.resize(nmax, 0.0);
  b_t.resize(nmax, 0.0);
  b_prc.resize(nmax, 0.0);
  b_prm.resize(nmax, 0.0);

  p_vec.resize(nmax, 0.0);
  q_vec.resize(nmax, 0.0);
  r_vec.resize(nmax, 0.0);
  d_vec.resize(nmax, 0.0);

  chi_mod.resize(nmax, 0.0);
  eta_mod.resize(nmax, 0.0);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::deallocate_storage()
{
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::reallocate_storage()
{
  deallocate_storage();
  allocate_storage();
  init_storage();
}

void FixQEqTersoff_k::init()
{
  if (!atom->q_flag) error->all(FLERR,"Fix qeq/tersoff_k requires atom attribute q");

  int irequest = neighbor->request(this);
  //neighbor->requests[irequest]->pair = 0;
  //neighbor->requests[irequest]->fix = 1;
  //neighbor->requests[irequest]->newton = 2;
  //neighbor->requests[irequest]->ghost = 1;

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::setup_pre_force(int vflag)
{
  //neighbor->build_one(list->index);

  deallocate_storage();
  allocate_storage();

  init_storage();

  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::min_setup_pre_force(int vflag)
{
  setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::init_storage()
{
  if(tersoff_k){
    tersoff_k->compute_pre_qeq(1,2);
    copy_pairdata();
    return;
  }
  int N = atom->nlocal + atom->nghost;
  for( int i = 0; i < N; i++ ) {
    Hdia_inv[i] = 1. / eta[atom->type[i]];
    b_s[i] = -chi[atom->type[i]];
    b_t[i] = -1.0;
    b_prc[i] = 0;
    b_prm[i] = 0;
    s[i] = t[i] = 0;
  }
}

void FixQEqTersoff_k::copy_pairdata()
{
  if(tersoff_k){
    int N = atom->nlocal + atom->nghost;
    for(int i = 0; i < N; i++){
      chi_mod[i] = eta_mod[i] = NAN;
    }
    for(int i = 0; i < atom->nlocal; i++){
      chi_mod[i] = tersoff_k->get_chi_i(i);
      eta_mod[i] = tersoff_k->get_bigj_ii(i);
    }

    for( int i = 0; i < N; i++ ) {
      Hdia_inv[i] = 1. / eta_mod[i];
      b_s[i] = -chi_mod[i];
      b_t[i] = -1.0;
      b_prc[i] = 0;
      b_prm[i] = 0;
      s[i] = t[i] = 0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::pre_force(int vflag)
{
  double t_start, t_end;

  if (update->ntimestep % nevery) return;
  if( comm->me == 0 ) t_start = MPI_Wtime();

  int n = atom->nlocal;
  int N = atom->nlocal + atom->nghost;
  // grow arrays if necessary
  // need to be atom->nmax in length
  if( atom->nmax > nmax ) reallocate_storage();

  tersoff_k->compute_pre_qeq(1,vflag);
  copy_pairdata();

  init_matvec();

  const sparse_matrix& A = tersoff_k->get_H();

  matvecs = CG(A, &(b_s[0]), &(s[0]));        // CG on s - parallel
  matvecs += CG(A, &(b_t[0]), &(t[0])); // CG on t - parallel
  calculate_Q();

  if( comm->me == 0 ) {
    t_end = MPI_Wtime();
    qeq_time = t_end - t_start;
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::pre_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::init_matvec()
{
  /* fill-in H matrix */
  int n = atom->nlocal;

  for( int i = 0; i < n; ++i ) {
    /* init pre-conditioner for H and init solution vectors */
    Hdia_inv[i] = 1. / eta_mod[i];
    b_s[i]      = -chi_mod[i];
    b_t[i]      = -1.0;

    /* linear extrapolation for s & t from previous solutions */
    //s[i] = 2 * s_hist[i][0] - s_hist[i][1];
    //t[i] = 2 * t_hist[i][0] - t_hist[i][1];

    /* quadratic extrapolation for s & t from previous solutions */
    //s[i] = s_hist[i][2] + 3 * ( s_hist[i][0] - s_hist[i][1] );
    t[i] = t_hist[i*nprev+2] + 3 * ( t_hist[i*nprev+0] - t_hist[i*nprev+1] );

    /* cubic extrapolation for s & t from previous solutions */
    s[i] = 4*(s_hist[i*nprev+0]+s_hist[i*nprev+2])-(6*s_hist[i*nprev+1]+s_hist[i*nprev+3]);
    //t[i] = 4*(t_hist[i][0]+t_hist[i][2])-(6*t_hist[i][1]+t_hist[i][3]);
  }

  pack_flag = 2;
  comm->forward_comm_fix(this); //Dist_vector( s );
  pack_flag = 3;
  comm->forward_comm_fix(this); //Dist_vector( t );
}

/* ---------------------------------------------------------------------- */

int FixQEqTersoff_k::CG(const sparse_matrix& A, double *b, double *x )
{
  int  i, j, imax;
  double tmp, alpha, beta, b_norm;
  double sig_old, sig_new, sig0;

  double* p = &(p_vec[0]);
  double* q = &(q_vec[0]);
  double* r = &(r_vec[0]);
  double* d = &(d_vec[0]);

  int n = atom->nlocal;
  imax = 200;

  pack_flag = 1;
  sparse_matvec( A, x, q );
  comm->reverse_comm_fix( this ); //Coll_Vector( q );

  vector_sum( r , 1.,  b, -1., q, n );
  for( j = 0; j < n; ++j )
    d[j] = r[j] * Hdia_inv[j]; //pre-condition

  b_norm = parallel_norm( b, n );
  sig_new = parallel_dot( r, d, n );
  sig0 = sig_new;

  for( i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i ) {
    comm->forward_comm_fix(this); //Dist_vector( d );
    sparse_matvec( A, d, q );
    comm->reverse_comm_fix(this); //Coll_vector( q );

    tmp = parallel_dot( d, q, n );
    alpha = sig_new / tmp;
    //  comm->me, i, parallel_norm( d, n ), parallel_norm( q, n ), tmp );

    vector_add( x, alpha, d, n );
    vector_add( r, -alpha, q, n );

    // pre-conditioning
    for( j = 0; j < n; ++j )
      p[j] = r[j] * Hdia_inv[j];

    sig_old = sig_new;
    sig_new = parallel_dot( r, p, n );


    beta = sig_new / sig_old;
    vector_sum( d, 1., p, beta, d, n );
  }

  if (i >= imax && comm->me == 0) {
    char str[128];
    sprintf(str,"Fix qeq/tersoff_k CG convergence failed after %d iterations "
            "at " BIGINT_FORMAT " step",i,update->ntimestep);
    error->warning(FLERR,str);
  }

  return i;
}


/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::sparse_matvec(const sparse_matrix &A, double *x, double *b )
{
  int i, j, itr_j;
  int n = atom->nlocal;
  int N = n + atom->nghost;

  for( i = 0; i < n; ++i )
    b[i] = eta_mod[i] * x[i];
  for( i = n; i < N; ++i )
    b[i] = 0;

  for( i = 0; i < n; ++i ) {
    for( itr_j=A.firstnbr[i]; itr_j<A.firstnbr[i]+A.numnbrs[i]; itr_j++) {
      j = A.jlist[itr_j];
      b[i] += A.val[itr_j] * x[j];
      b[j] += A.val[itr_j] * x[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::calculate_Q()
{
  int i, k;
  double u, s_sum, t_sum;
  double *q = atom->q;
  int n = atom->nlocal;

  s_sum = parallel_vector_acc( &(s[0]), n );
  t_sum = parallel_vector_acc( &(t[0]), n);
  u = s_sum / t_sum;

  for( i = 0; i < n; ++i ) {
    if(i < 100){
    }
    q[i] = s[i] - u * t[i];

    /* backup s & t */
    for( k = 4; k > 0; --k ) {
      s_hist[i*nprev+k] = s_hist[i*nprev+k-1];
      t_hist[i*nprev+k] = t_hist[i*nprev+k-1];
    }
    s_hist[i*nprev+0] = s[i];
    t_hist[i*nprev+0] = t[i];
  }

  pack_flag = 4;
  comm->forward_comm_fix( this ); //Dist_vector( atom->q );
}

/* ---------------------------------------------------------------------- */

int FixQEqTersoff_k::pack_forward_comm(int n, int *list, double *buf,
                          int pbc_flag, int *pbc)
{
  int m;

  if( pack_flag == 1)
    for(m = 0; m < n; m++) buf[m] = d_vec[list[m]];
  else if( pack_flag == 2 )
    for(m = 0; m < n; m++) buf[m] = s[list[m]];
  else if( pack_flag == 3 )
    for(m = 0; m < n; m++) buf[m] = t[list[m]];
  else if( pack_flag == 4 )
    for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];

  return n;
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if( pack_flag == 1)
    for(m = 0, i = first; m < n; m++, i++) d_vec[i] = buf[m];
  else if( pack_flag == 2)
    for(m = 0, i = first; m < n; m++, i++) s[i] = buf[m];
  else if( pack_flag == 3)
    for(m = 0, i = first; m < n; m++, i++) t[i] = buf[m];
  else if( pack_flag == 4)
    for(m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
}

/* ---------------------------------------------------------------------- */

int FixQEqTersoff_k::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  for(m = 0, i = first; m < n; m++, i++) buf[m] = q_vec[i];
  return n;
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::unpack_reverse_comm(int n, int *list, double *buf)
{
  for(int m = 0; m < n; m++) q_vec[list[m]] += buf[m];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixQEqTersoff_k::memory_usage()
{
  double bytes;

  bytes = atom->nmax*nprev*2 * sizeof(double); // s_hist & t_hist
  bytes += atom->nmax*11 * sizeof(double); // storage

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqTersoff_k::grow_arrays(int nmax)
{
  //memory->grow(s_hist,nmax,nprev,"qeq:s_hist");
  //memory->grow(t_hist,nmax,nprev,"qeq:t_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqTersoff_k::copy_arrays(int i, int j, int delflag)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j*nprev+m] = s_hist[i*nprev+m];
    t_hist[j*nprev+m] = t_hist[i*nprev+m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixQEqTersoff_k::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i*nprev+m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i*nprev+m];
  return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixQEqTersoff_k::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal*nprev+m] = buf[m];
  for (int m = 0; m < nprev; m++) t_hist[nlocal*nprev+m] = buf[nprev+m];
  return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixQEqTersoff_k::parallel_norm( double *v, int n )
{
  int  i;
  double my_sum, norm_sqr;

  my_sum = 0;
  for( i = 0; i < n; ++i )
    my_sum += v[i]*v[i];

  MPI_Allreduce( &my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world );

  return sqrt( norm_sqr );
}

/* ---------------------------------------------------------------------- */

double FixQEqTersoff_k::parallel_dot( double *v1, double *v2, int n )
{
  int  i;
  double my_dot, res;

  my_dot = 0;
  res = 0;
  for( i = 0; i < n; ++i )
    my_dot += v1[i] * v2[i];

  MPI_Allreduce( &my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world );

  return res;
}

/* ---------------------------------------------------------------------- */

double FixQEqTersoff_k::parallel_vector_acc( double *v, int n )
{
  int  i;
  double my_acc, res;

  my_acc = 0;
  for( i = 0; i < n; ++i )
    my_acc += v[i];

  MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world );

  return res;
}

/* ---------------------------------------------------------------------- */

double FixQEqTersoff_k::norm( double* v1, int k )
{
  double ret = 0;

  for( --k; k>=0; --k )
    ret +=  ( v1[k] * v1[k] );

  return sqrt( ret );
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::vector_sum( double* dest, double c, double* v,
                                double d, double* y, int k )
{
  for( --k; k>=0; --k )
    dest[k] = c * v[k] + d * y[k];
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::vector_scale( double* dest, double c, double* v, int k )
{
  for( --k; k>=0; --k )
    dest[k] = c * v[k];
}

/* ---------------------------------------------------------------------- */

double FixQEqTersoff_k::dot( double* v1, double* v2, int k )
{
  double ret = 0;

  for( --k; k>=0; --k )
    ret +=  v1[k] * v2[k];

  return ret;
}

/* ---------------------------------------------------------------------- */

void FixQEqTersoff_k::vector_add( double* dest, double c, double* v, int k )
{
  for( --k; k>=0; --k )
    dest[k] += c * v[k];
}
