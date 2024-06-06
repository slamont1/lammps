/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// #include "fix_volume_voronoi_global.h"

// #include "arg_info.h"
// #include "atom.h"
// #include "atom_masks.h"
// #include "comm.h"
// #include "compute.h"
// #include "domain.h"
// #include "error.h"
// #include "force.h"
// #include "group.h"
// #include "input.h"
// #include "math_extra.h"
// #include "memory.h"
// #include "modify.h"
// #include "output.h"
// #include "region.h"
// #include "respa.h"
// #include "update.h"
// #include "variable.h"

// #include <algorithm>
// #include <bits/stdc++.h>
// #include <fstream>
// #include <iostream>
// #include <chrono>
// #include <cmath>
// #include <cstring>
// #include <cstdlib>
// #include <cstdio>
// #include <math.h>
// #include <random>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string>
// #include <vector>

// using namespace LAMMPS_NS;
// using namespace FixConst;
// using namespace std;
// using namespace std::chrono;
// using namespace voro;

// enum { NONE, CONSTANT, EQUAL, ATOM };

// /* ---------------------------------------------------------------------- */

// FixVolumeVoronoiGlobal::FixVolumeVoronoiGlobal(LAMMPS *lmp, int narg, char **arg) :
//     Fix(lmp, narg, arg), id_compute_voronoi(nullptr), total_virial(nullptr)
// {
//   if (narg < 6) error->all(FLERR, "Illegal fix volume/voronoi/global command: not sufficient args");

//   MPI_Comm_rank(world,&me);
//   MPI_Comm_size(world, &nprocs);

//   // In case atoms change groups
//   dynamic_group_allow = 1;

//   // For virial pressure contributions
//   virial_global_flag  = 1;
//   thermo_virial = 1;

//   // For global energy contribution
//   scalar_flag = 1;
//   global_freq = 1;
//   extscalar = 1;
//   energy_global_flag = 1;
//   thermo_energy = 1;

//   // Parse first two arguments: elasticity and preferred area
//   eps_a  = utils::numeric(FLERR,arg[3],false,lmp);
//   cdens0 = utils::numeric(FLERR,arg[4],false,lmp);
//   f0     = utils::numeric(FLERR,arg[5],false,lmp);

//   // Default nevery for now
//   nevery = 1;

//   // Countflag to only initialize once
//   countflag = 0;

//   // Initialize energy counting
//   eflag = 0;
//   evoro = 0.0;

// }

// /* ---------------------------------------------------------------------- */

// FixVolumeVoronoiGlobal::~FixVolumeVoronoiGlobal()
// {
  
// }

// /* ---------------------------------------------------------------------- */
// // returntype classname :: functidentifier(args) }

// int FixVolumeVoronoiGlobal::setmask()
// {
//   datamask_read = datamask_modify = 0;

//   int mask = 0;
//   mask |= PRE_FORCE;
//   mask |= POST_FORCE;
//   return mask;
// }

// /* ---------------------------------------------------------------------- */

// void FixVolumeVoronoiGlobal::init()
// {
//   // Special weights must all be one to ensure 1-3 and 1-4 neighbors are skipped
//   if (force->special_lj[1] != 1.0 || force->special_lj[2] != 1.0 || force->special_lj[3] != 1.0) {
//     error->all(FLERR,"Fix volume/voronoi/global requires all special weights to be 1.0");
//   }
//   if (force->special_coul[1] != 1.0 || force->special_coul[2] != 1.0 || force->special_coul[3] != 1.0) {
//     error->all(FLERR,"Fix volume/voronoi/global requires all special weights to be 1.0");
//   }

//   // Newton_bond must be off for all bonds to be seen
//   if (force->newton_bond) {
//     error->all(FLERR,"Fix volume/voronoi/global requires newton_bond off");
//   }

//   // Newton_pair must be on because I haven't tested it without it lol
//   if (!force->newton_pair) {
//     error->all(FLERR,"Fix volume/voronoi/global requires newton_pair on");
//   }

//   // Only works for 2D simulations at the moment
//   if (domain->dimension == 3) {
//     error->all(FLERR,"Fix volume/voronoi/global does not support 3D simulations");
//   }

//   // Thickness in z-dimension must be 1 for volume calculation to be valid
//   if (domain->zprd != 1.0) {
//     error->all(FLERR,"Fix volume/voronoi/global requires z-thickness to be 1.0");
//   }

// }


// /* ---------------------------------------------------------------------- */

// void FixVolumeVoronoiGlobal::setup(int vflag)
// {

//   // Only perform energy calculations if already invoked
//   if (countflag) {
//     // Initialize energetic quantities
//     pre_force(vflag);
//     post_force(vflag);
//     return;
//   }
//   countflag = 1;

//   voro_volume0 = domain->xprd * domain->yprd;

//   // Run single timestep
//   pre_force(vflag);
//   post_force(vflag);

// }

// /* ---------------------------------------------------------------------- */

// void FixVolumeVoronoiGlobal::pre_force(int vflag)
// {

//   // Reset energy counter
//   eflag = 0;
//   evoro = 0.0;

//   // For future implementation (nevery)
//   if (update->ntimestep % nevery) return;

//   // Current system volume
//   double voro_volume = domain->xprd * domain->yprd;

//   // Critical volume
//   double volume_crit = f0*voro_volume0;

//   // Number of particles
//   double np = cdens0*voro_volume0;

//   // Calculate pressure
//   pressure = 0.0;
//   pressure  = 1.0/(voro_volume - f0*voro_volume0);
//   pressure -= eps_a*f0*voro_volume0/voro_volume/voro_volume;
//   pressure *= -np;

//   // Tally energy
//   double etmp = 0.0;
//   etmp += log(voro_volume - f0*voro_volume0);
//   etmp += eps_a*f0*voro_volume0/voro_volume;
//   etmp *= -np;
//   evoro += etmp;
// }

// /* ---------------------------------------------------------------------- */

// void FixVolumeVoronoiGlobal::post_force(int vflag)
// {

//   // virial setup
//   v_init(vflag);

//   // Tally virial (only 11 and 22 components)
//   v_tally(0,0,pressure);
//   v_tally(1,0,pressure);

// }

// double FixVolumeVoronoiGlobal::compute_scalar()
// {
//   // only sum across procs one time

//   if (eflag == 0) {
//     MPI_Allreduce(&evoro,&evoro_all,1,MPI_DOUBLE,MPI_SUM,world);
//     eflag = 1;
//   }
//   return evoro_all;
// }