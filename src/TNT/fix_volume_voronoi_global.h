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

// #ifdef FIX_CLASS
// // clang-format off
// FixStyle(volume/voronoi/global,FixVolumeVoronoiGlobal);
// // clang-format on
// #else

// //PRB{These Statements ensire that this class is only inlcuded once in the project}
// #ifndef LMP_FIX_VOLUME_VORONOI_GLOBAL_H
// #define LMP_FIX_VOLUME_VORONOI_GLOBAL_H

// #include "fix.h"

// namespace LAMMPS_NS {

// class FixVolumeVoronoiGlobal : public Fix {
//  public:
//   FixVolumeVoronoiGlobal(class LAMMPS *, int, char **);
//   ~FixVolumeVoronoiGlobal() override;
//   int setmask() override;
//   void init() override;
//   void setup(int) override;
//   void pre_force(int) override;
//   void post_force(int) override;
//   double compute_scalar() override;
  
//   FILE *fp;

//  protected:
//    int me, nprocs; 
//    int nmax;

//  private:

//   // For global pressure
//   double pressure;

//   // For global energy
//   int eflag;
//   double evoro, evoro_all;

//   // Initial simulation area
//   double voro_volume0;

//   // For fluid bulk resistance
//   double eps_a, cdens0, f0;

//   // For keeping track of invoking DT creation
//   int countflag;

// };

// }    // namespace LAMMPS_NS

// #endif
// #endif
