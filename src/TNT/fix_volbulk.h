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
FixStyle(volbulk,FixVolBulk);
// clang-format on
#else

//PRB{These Statements ensire that this class is only inlcuded once in the project}
#ifndef LMP_FIX_VOLBULK_H
#define LMP_FIX_VOLBULK_H

#include "fix.h"

namespace LAMMPS_NS {

class FixVolBulk : public Fix {
 public:
  FixVolBulk(class LAMMPS *, int, char **);
  ~FixVolBulk() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;
  
  FILE *fp;

 protected:
 int me, nprocs; 
 int nmax;

 private:
  
  class Compute *vcompute; // ptr to compute voronoi
  class Fix *fstore;       // ptr to fix/store

  double Elasticity, VolPref;
  int flag_store_init;
 
  char *id_compute_voronoi, *id_fix_store;

  int ilevel_respa;

  // To read peratom volume from compute voronoi
  double *voro_volume;
  double *voro_volume0;

  // Total virial on each cell
  double **total_virial;

  // For communicating voro_volume or voro_volume0
  int commflag;

};

}    // namespace LAMMPS_NS

#endif
#endif
