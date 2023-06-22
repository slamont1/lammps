/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(customforce,FixCustomForce);
// clang-format on
#else

//PRB{These Statements ensire that this class is only inlcuded once in the project}
#ifndef LMP_FIX_CUSTOMFORCE_H
#define LMP_FIX_CUSTOMFORCE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixCustomForce : public Fix {
 public:
  FixCustomForce(class LAMMPS *, int, char **);
  ~FixCustomForce() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  
  FILE *fp;

 protected:
 int me, nprocs; 

 private:
  
  double Elasticity, Apref;
  int  nvalues;
  int *which, *argindex, *value2index;
  char **ids;
 
  char *idregion;
  class Region *region;
 
  int ilevel_respa;

 // To read peratom data from compute voronoi
  double **voro_data;
  int commflag;

};

}    // namespace LAMMPS_NS

#endif
#endif
