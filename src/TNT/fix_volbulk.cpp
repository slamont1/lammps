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

#include "fix_volbulk.h"

#include "arg_info.h"
#include "atom.h"
#include "atom_masks.h"
#include "cell.hh"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "input.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace std::chrono;
using namespace voro;

enum { NONE, CONSTANT, EQUAL, ATOM };

/* ---------------------------------------------------------------------- */

FixVolBulk::FixVolBulk(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), voro_volume(nullptr), voro_volume0(nullptr), id_compute_voronoi(nullptr), total_virial(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal fix volbulk command: not sufficient args");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world, &nprocs);

  dynamic_group_allow = 1;
  energy_peratom_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_energy = thermo_virial = 1;

  respa_level_support = 1;
  ilevel_respa = 0;

  // Parse first two arguments: elasticity and preferred area
  Elasticity = utils::numeric(FLERR,arg[3],false,lmp);
  VolPref = utils::numeric(FLERR,arg[4],false,lmp);

  // Parse compute id
  id_compute_voronoi = utils::strdup(arg[5]);
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute) error->all(FLERR,"Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  // parse values for optional args
  flag_store_init = 0;
  id_fix_store = nullptr;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"store_init") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix voronoi command");
      flag_store_init = 1;

      id_fix_store = utils::strdup(arg[iarg+1]);
      fstore = modify->get_fix_by_id(id_fix_store);
      if (!fstore) error->all(FLERR,"Could not find fix ID {} for voronoi fix/store", id_fix_store);

      iarg += 2;
    } else error->all(FLERR,"Illegal fix voronoi command");
  }

  nevery = 1;

  nmax = atom->nmax;
  voro_volume = nullptr;
  voro_volume0 = nullptr;
  total_virial = nullptr;

}

/* ---------------------------------------------------------------------- */

FixVolBulk::~FixVolBulk()
{
  delete[] id_compute_voronoi;
  delete[] id_fix_store;
  
  memory->destroy(voro_volume);
  memory->destroy(total_virial);
  if (flag_store_init) memory->destroy(voro_volume0);
  
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) }

int FixVolBulk::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolBulk::init()
{
  // set indices and check validity of all computes and variables

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}


/* ---------------------------------------------------------------------- */

void FixVolBulk::setup(int vflag)
{

  memory->create(voro_volume,nmax,"volbulk:voro_volume");
  memory->create(total_virial,nmax,6,"volbulk:total_virial");
  if (flag_store_init) memory->create(voro_volume0,nmax,"volbulk:voro_volume0");

  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }

}

/* ---------------------------------------------------------------------- */

void FixVolBulk::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolBulk::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  double xn[3], yn[3];
  double x0, x1, x2, y0, y1, y2;

  // For future implementation (nevery)
  if (update->ntimestep % nevery) return;

  // virial setup

  v_init(vflag);

  int me = comm->me;  //current rank value

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(voro_volume);
    memory->destroy(total_virial);
    if (flag_store_init) memory->destroy(voro_volume0);
    nmax = atom->nmax;
    memory->create(voro_volume,nmax,"volbulk:voro_volume");
    memory->create(total_virial,nmax,6,"volbulk:voro_volume");
    if (flag_store_init) memory->create(voro_volume0,nmax,"volbulk:voro_volume0");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    voro_volume[i] = 0.0;
    if (flag_store_init) voro_volume0[i] = 0.0;
    for (int j = 0; j < 5; j++) {
        total_virial[i][j] = 0.0;
    }
  }

  // Invoke compute
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
    vcompute->compute_peratom();
    vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // Define pointer to fix_store
  if (flag_store_init) fstore = modify->get_fix_by_id(id_fix_store);

  // Fill voro_volume and voro_volume0 with values from compute voronoi
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) voro_volume[i] = vcompute->array_atom[i][0];
    if (flag_store_init) {
      if (mask[i] & groupbit) voro_volume0[i] = fstore->vector_atom[i];
    }
  }
  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // DON'T NEED TO COMMUNICATE THIS ANYMORE
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  // forward communication of voronoi data:
//   commflag = 0;
//   comm->forward_comm(this,1);

  // forward communication of initial cell areas:
//   if (flag_store_init) {
//     commflag = 1;
//     comm->forward_comm(this,1);
//   }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  // Determine the number of rows in the local array for voronoi data
  int numRows = vcompute->size_local_rows;

  // Loop through each cell
  double rvec[3],uvec[3],fvec[3];
  for (int n = 0; n < numRows; n++) {

    // Atom that owns this cell
    int i = atom->map(vcompute->array_local[n][0]);
    if (i >= nlocal) {
        printf("Skipping atom %d \n",i);
        continue;
    }
    if (i < 0)
      error->one(FLERR,"Can't find local atom");

    // Skip external faces (box boundaries)
    if (vcompute->array_local[n][1] == 0.0) continue;

    // Atom that shares this cell edge
    int j = atom->map(vcompute->array_local[n][1]);
    if (j < 0) {
      printf("Found local id %d from global tag %f and owned cell local id %d global id %f \n",j,vcompute->array_local[n][1],i,vcompute->array_local[n][0]);
      error->one(FLERR,"Fix volbulk needs ghost atoms from further away");
    }

    // Area of this cell edge
    double edge_area = vcompute->array_local[n][2];

    // Calculate pressure due to volume change
    double volume = voro_volume[i];
    double pressure = 0.0;
    if (flag_store_init) {
      double VolPref0 = voro_volume0[i];
      pressure = Elasticity*VolPref0*(volume-VolPref0);
    } else {
      pressure = Elasticity*VolPref*(volume-VolPref);
    }

    // Calculate net force acting on particle j
    double fnet = pressure*edge_area;

    // Calculate vectors spanning between particle i and j
    rvec[0] = x[j][0]-x[i][0];
    rvec[1] = x[j][1]-x[i][1];
    rvec[2] = x[j][2]-x[i][2];
    domain->minimum_image(rvec[0], rvec[1], rvec[2]);
    double rmag = pow(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2],0.5);
    uvec[0] = rvec[0]/rmag;
    uvec[1] = rvec[1]/rmag;
    uvec[2] = rvec[2]/rmag;

    // Force vector due to volume change
    fvec[0] = fnet*uvec[0];
    fvec[1] = fnet*uvec[1];
    fvec[2] = fnet*uvec[2];

    // Add force to particle j
    f[j][0] += fvec[0];
    f[j][1] += fvec[1];
    f[j][2] += fvec[2];

    // Add virial contribution to particle j
    total_virial[j][0] -= fvec[0]*rvec[0];
    total_virial[j][1] -= fvec[1]*rvec[1];
    total_virial[j][2] -= fvec[2]*rvec[2];
    total_virial[j][3] -= fvec[0]*rvec[1];
    total_virial[j][4] -= fvec[0]*rvec[2];
    total_virial[j][5] -= fvec[1]*rvec[2];

    // // Add force to particle i ?????
    // f[i][0] -= fvec[0];
    // f[i][1] -= fvec[1];
    // f[i][2] -= fvec[2];

    // // Add virial contribution to particle i ???????????????
    // total_virial[i][0] += fvec[0]*rvec[0];
    // total_virial[i][1] += fvec[1]*rvec[1];
    // total_virial[i][2] += fvec[2]*rvec[2];
    // total_virial[i][3] += fvec[0]*rvec[1];
    // total_virial[i][4] += fvec[0]*rvec[2];
    // total_virial[i][5] += fvec[1]*rvec[2];

  }

  // Reverse communication of force vectors
  comm->reverse_comm();

  // Reverse communication of virial contributions
  comm->reverse_comm(this);

  // Tally virial contributions of owned atoms
  for (int i = 0; i < nlocal; i++) {
    if (evflag) {
      v_tally(i, total_virial[i]);
    }
  }
 
}

/* ---------------------------------------------------------------------- */

void FixVolBulk::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolBulk::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

int FixVolBulk::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)

{

  int i,j,m;

  m = 0;
 if(commflag == 1){
    for(i = 0; i < n; i++){
      j = list[i];
      buf[m++] = voro_volume0[j];
    }
 } else {
    for(i = 0; i < n; i++){
      j = list[i];
      buf[m++] = voro_volume[j];
    }
  }
  return m;
}

void FixVolBulk::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

 if (commflag == 1) {
    for (i = first; i < last; i++){
      voro_volume0[i] = buf[m++];
    }
 } else {
    for (i = first; i < last; i++){
      voro_volume[i] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixVolBulk::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    for (int v = 0; v < 5; v++) {
        buf[m++] = total_virial[i][v];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixVolBulk::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    for (int v = 0; v < 5; v++) {
      total_virial[j][v] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixVolBulk::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax*8 * sizeof(double);
  return bytes;
}
