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

#include "fix_volvoro.h"

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

FixVolVoro::FixVolVoro(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), id_compute_voronoi(nullptr), total_virial(nullptr)
{
  if (narg < 6) error->all(FLERR, "Illegal fix volvoro command: not sufficient args");

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

  // Max faces of a voronoi cell
  max_faces = utils::inumeric(FLERR,arg[6],false,lmp);

  // parse values for optional args
  flag_store_init = 0;
  id_fix_store = nullptr;

  int iarg = 7;
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

  countflag = 0;
  nmax = atom->nmax;
  total_virial = nullptr;

}

/* ---------------------------------------------------------------------- */

FixVolVoro::~FixVolVoro()
{
  delete[] id_compute_voronoi;
  delete[] id_fix_store;
  
  memory->destroy(total_virial);

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;
  
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) }

int FixVolVoro::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_constructor()
{

  // Create call to fix property/atom for storing DT faces
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom i2_dtf_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(max_faces)));

  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("dtf_")+id),tmp1,tmp2);

  tagint **dtf = atom->iarray[index];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // Initialize entries to 0
  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < max_faces; m++) {
      if (mask[i] & groupbit) {
        dtf[i][m] = 0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::init()
{
  // set indices and check validity of all computes and variables

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}


/* ---------------------------------------------------------------------- */

void FixVolVoro::setup(int vflag)
{

  // Return if this has already been invoked
  if (countflag) return;
  countflag = 1;

  // Ensure that computes have been invoked
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
    vcompute->compute_peratom();
    vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // Pointer to dtf
  tagint **dtf = atom->iarray[index];

  // Determine the number of rows in the local array for voronoi data
  int numRows = vcompute->size_local_rows;

  // Loop through numRows and fill dtf
  for (int n = 0; n < numRows; n++) {

    // Skip external faces (box boundaries) and empty cells
    if (vcompute->array_local[n][1] == 0.0) continue;

    // Atom that owns this cell
    int i = atom->map(vcompute->array_local[n][0]);
    if (i < 0)
      error->one(FLERR,"Can't find local atom");

    // Atom that shares this cell edge
    int j = atom->map(vcompute->array_local[n][1]);
    if (j < 0) {
      printf("Found local id %d from global tag %f and owned cell local id %d global id %f \n",j,vcompute->array_local[n][1],i,vcompute->array_local[n][0]);
      error->one(FLERR,"Fix volbulk needs ghost atoms from further away");
    }
    tagint tagj = atom->tag[j];

    // Find next empty space for atom i and fill it with tagj
    for (int m = 0; m < max_faces; m++) {
        if (dtf[i][m] == 0) {
            dtf[i][m] = tagj;
            break;
        }
    }
  }

  // Confirm cyclic permutation of dtf
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    arrange_cyclic(dtf[i],i);
  }

  // Communicate dtf
//   commflag = 2;
  comm->forward_comm(this,max_faces);
    
  // Create memory allocations
  nmax = atom->nmax;
  memory->create(total_virial,nmax,6,"volvoro:total_virial");

  if (utils::strmatch(update->integrate_style, "^verlet"))
      post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
      post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }

}

/* ---------------------------------------------------------------------- */

void FixVolVoro::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_integrate()
{
  // In post integrate:
  // 1. Compute triangulation from compute voro
  //    - Only need to do this once
  // 2. (Later) flip triangulation edges
  return;

}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  tagint **dtf = atom->iarray[index];

  // For future implementation (nevery)
  if (update->ntimestep % nevery) return;

  // virial setup
  v_init(vflag);

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(total_virial);
    nmax = atom->nmax;
    memory->create(total_virial,nmax,6,"volvoro:total_virial");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
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

  // Loop through local atoms:
  // 1. Apply force `on i from i'
  // 2. Apply force `on j from i'
  // 3. Apply force `on k from i'
  for (int i = 0; i < nlocal; i++) {

    // Volume and pressure due to volume change
    double voro_volume = vcompute->array_atom[i][0];
    double voro_volume0;
    if (flag_store_init) {
        voro_volume0 = fstore->vector_atom[i];
    } else {
        voro_volume0 = VolPref;
    }
    double pressure = Elasticity*(voro_volume-voro_volume0)*0.5;

    // Declare variables inside loop for scope clarity
    double fx,fy;
    int j,jtmp,num_faces,jleft,jright;
    double x1,x2,y1,y2;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~ Refresh vector arrays ~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // Local ids of atoms in current triangulation
    int DT[3] = {0};

    // Coordinates of current, previous, and next vertex
    double vert[2] = {0.0}, vert_prev[2] = {0.0}, vert_next[2] = {0.0};

    // x and y coordinates of current triangulation
    double xn[3] = {0.0}, yn[3] = {0.0};

    // Vector spanning between n+1 and n-1 vertices
    double rnu_diff[2] = {0.0};

    // Vector spanning between the vertex and the atom
    double rvec[2] = {0.0};

    // Jacobian matrix (Voigt notation): J11, J22, J12, J21
    double Jac[4] = {0.0};

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~ Begin calculations ~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // First entry of DT is always i
    DT[0] = i;

    // Find num_faces for this atom
    for (int n = 0; n < max_faces; n++) {
        if (dtf[i][n] == 0) {
            num_faces = n;
            break;
        }
    }

    // Coords of atom i
    double x0 = x[i][0];
    double y0 = x[i][1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~ Calculate vertices ~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // Local id and coords of last face
    jtmp = atom->map(dtf[i][num_faces-1]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    x1 = x[j][0];
    y1 = x[j][1];

    // Local id and coords of first face
    jtmp = atom->map(dtf[i][0]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    DT[1] = j;
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of previous vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert_prev);
    x1 = x2;
    y1 = y2;

    // Local id and coords of second face
    jtmp = atom->map(dtf[i][1]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    DT[2] = j;
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of current vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert);
    x1 = x2;
    y1 = y2;

    // Local id and coords of third face
    jtmp = atom->map(dtf[i][2]);
    if (jtmp < 0) {
      printf("\n local atom %d with tag %d looking for atom with tag %d \n",i,atom->tag[i],dtf[i][2]);
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of next vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert_next);
    x1 = x2;
    y1 = y2;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on cell i ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    // rnu_diff is inverted due to cross product with outward normal
    rnu_diff[0] = vert_next[1] - vert_prev[1];
    rnu_diff[1] = -(vert_next[0] - vert_prev[0]);

    // Calculate jacobian on atom i (cell owner)
    Jacobian(DT,i,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[i][0] += fx;
    f[i][1] += fy;

    // Vector for virial calculation
    rvec[0] = vert[0] - x[i][0];
    rvec[1] = vert[1] - x[i][1];

    // Virial contributions
    total_virial[i][0] += fx*rvec[0];
    total_virial[i][1] += fy*rvec[1];
    total_virial[i][3] += fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on jleft ~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    jleft = DT[1];

    // Calculate jacobian on atom jleft
    Jacobian(DT,jleft,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[jleft][0] += fx;
    f[jleft][1] += fy;

    // Vector for virial calculation
    rvec[0] = vert[0] - x[jleft][0];
    rvec[1] = vert[1] - x[jleft][1];

    // Virial contributions
    total_virial[jleft][0] -= fx*rvec[0];
    total_virial[jleft][1] -= fy*rvec[1];
    total_virial[jleft][3] -= fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on jright ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    jright = DT[2];

    // Calculate jacobian on atom jright
    Jacobian(DT,jright,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[jright][0] += fx;
    f[jright][1] += fy;

    // Vector for virial calculation
    rvec[0] = vert[0] - x[jright][0];
    rvec[1] = vert[1] - x[jright][1];

    // Virial contributions
    total_virial[jright][0] -= fx*rvec[0];
    total_virial[jright][1] -= fy*rvec[1];
    total_virial[jright][3] -= fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Permute vertices cyclically ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    // vert becomes previous vert
    vert_prev[0] = vert[0];
    vert_prev[1] = vert[1];

    // Next vert becomes vert
    vert[0] = vert_next[0];
    vert[1] = vert_next[1];

    // Second entry of DT becomes jright
    DT[1] = jright;

    // Loop through faces
    for (int n = 1; n < num_faces; n++) {

        // Only need to find the index of the n+2 face
        int jnext;
        if (n + 2 == num_faces) {

            jtmp = atom->map(dtf[i][0]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][num_faces-1]);
            DT[2] = domain->closest_image(i,jtmp);
        } else if (n + 1 == num_faces) {

            jtmp = atom->map(dtf[i][1]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][0]);
            DT[2] = domain->closest_image(i,jtmp);
        } else {

            jtmp = atom->map(dtf[i][n+2]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][n+1]);
            DT[2] = domain->closest_image(i,jtmp);
        }

        // Find circumcenter of next vertex
        x2 = x[jnext][0];
        y2 = x[jnext][1];
        xn[0] = x0; xn[1] = x1; xn[2] = x2;
        yn[0] = y0; yn[1] = y1; yn[2] = y2;
        calc_cc(xn, yn, vert_next);
        x1 = x2;
        y1 = y2;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on cell i ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        // rnu_diff is inverted due to cross product with outward normal
        rnu_diff[0] = vert_next[1] - vert_prev[1];
        rnu_diff[1] = -(vert_next[0] - vert_prev[0]);

        // Calculate jacobian on atom i (cell owner)
        Jacobian(DT,i,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[i][0] += fx;
        f[i][1] += fy;

        // Vector for virial calculation
        rvec[0] = vert[0] - x[i][0];
        rvec[1] = vert[1] - x[i][1];

        // Virial contributions
        total_virial[i][0] += fx*rvec[0];
        total_virial[i][1] += fy*rvec[1];
        total_virial[i][3] += fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on jleft ~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        jleft = DT[1];

        // Calculate jacobian on atom jleft
        Jacobian(DT,jleft,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[jleft][0] += fx;
        f[jleft][1] += fy;

        // Vector for virial calculation
        rvec[0] = vert[0] - x[jleft][0];
        rvec[1] = vert[1] - x[jleft][1];

        // Virial contributions
        total_virial[jleft][0] -= fx*rvec[0];
        total_virial[jleft][1] -= fy*rvec[1];
        total_virial[jleft][3] -= fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on jright ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        jright = DT[2];

        // Calculate jacobian on atom jleft
        Jacobian(DT,jright,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[jright][0] += fx;
        f[jright][1] += fy;

        // Vector for virial calculation
        rvec[0] = vert[0] - x[jright][0];
        rvec[1] = vert[1] - x[jright][1];

        // Virial contributions
        total_virial[jright][0] -= fx*rvec[0];
        total_virial[jright][1] -= fy*rvec[1];
        total_virial[jright][3] -= fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Permute vertices cyclically ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        // vert becomes previous vert
        vert_prev[0] = vert[0];
        vert_prev[1] = vert[1];

        // Next vert becomes vert
        vert[0] = vert_next[0];
        vert[1] = vert_next[1];

        // Second entry of DT becomes jright
        DT[1] = jright;

    }
  }

  // Reverse communication of force vectors
  comm->reverse_comm();

  // Reverse communication of virial contributions
  comm->reverse_comm(this,6);

  // Tally virial contributions of owned atoms
  for (int i = 0; i < nlocal; i++) {
    if (evflag) {
      v_tally(i, total_virial[i]);
    }
  }
 
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

void FixVolVoro::arrange_cyclic(tagint *tag_vec, int icell)
{
  double **x = atom->x;
  int num_faces = 0;
  int jLeft;

  // Find bottom-left point
  jLeft = domain->closest_image(icell,atom->map(tag_vec[0]));
  for (int i = 0; i < max_faces; i++) {

    // Break if there are no more faces
    if (tag_vec[i] == 0) break;

    // Local id of current face
    int jtmp = atom->map(tag_vec[i]);
    if (jtmp < 0)
      error->one(FLERR,"Fix bond/dynamic needs ghost atoms "
                "from further away");
    int j = domain->closest_image(icell,jtmp);

    // Add to the number of faces
    num_faces += 1;

    // Determine if this point is lower than bottomLeft
    if (x[j][1] < x[jLeft][1] || (x[j][1] == x[jLeft][1] && x[j][0] < x[jLeft][0]) ) {
        jLeft = j;
    }
  }

  // Calculate angle for all points
  double theta[num_faces] = {0.0};
  int indices[num_faces];
  for (int i = 0; i < num_faces; i++) {

    // Local id of current face
    int jtmp = atom->map(tag_vec[i]);
    int j = domain->closest_image(icell,jtmp);
    theta[i] = atan2(x[j][1] - x[jLeft][1],x[j][0] - x[jLeft][0]);

    // indices array for sorting
    indices[i] = i;
  }

  // Sort indices based on minimum angle
  std::sort(indices, indices + num_faces, [&](int i, int n) { return theta[i] < theta[n]; });

  // Create dummy vector for sorting
  tagint tag_sorted[max_faces];
  for (int i = 0; i < max_faces; i++) {
    tag_sorted[i] = 0;
  }
  for (int i = 0; i < num_faces; i++) {
    tag_sorted[i] = tag_vec[indices[i]];
  }

  // Fill tag_vec with sorted values
  for (int i = 0; i < num_faces; i++) {
    tag_vec[i] = tag_sorted[i];
  }

}

/*------------------------------------------------------------------------*/

void FixVolVoro::calc_cc(double *xn, double *yn,  double *CC)
{
  double x0 = xn[0];
  double x1 = xn[1];
  double x2 = xn[2];

  double y0 = yn[0];
  double y1 = yn[1];
  double y2 = yn[2];

  double a = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
  double b = sqrt(pow(x2-x0,2)+pow(y2-y0,2));
  double c = sqrt(pow(x1-x0,2)+pow(y1-y0,2));

 // Find Angles A, B and C

  double A = acos((b*b+c*c-a*a)/(2*b*c));
  double B = acos((a*a+c*c-b*b)/(2*a*c));
  double C = acos((a*a+b*b-c*c)/(2*a*b));

  CC[0] = (x0*sin(2*A)+x1*sin(2*B)+x2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // x coord of circumcenter
  CC[1] = (y0*sin(2*A)+y1*sin(2*B)+y2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // y coord of circumcenter
}

/*------------------------------------------------------------------------*/

void FixVolVoro::Jacobian(int *DT, int p, double *Jac){

  double **x = atom->x;

  // Identify j and k
  int j,k;
  if (DT[0] == p) {
      j = DT[1];
      k = DT[2];
  }
  else if (DT[1] == p){
      j = DT[0];
      k = DT[2];
  }
  else if (DT[2] == p){
      j = DT[0];
      k = DT[1];
  }

  double ri[3] = {x[p][0], x[p][1], 0};
  double rj[3] = {x[j][0], x[j][1], 0};
  double rk[3] = {x[k][0], x[k][1], 0};

  double li = sqrt(pow(rj[0]-rk[0],2)+pow(rj[1]-rk[1],2));
  double lj = sqrt(pow(ri[0]-rk[0],2)+pow(ri[1]-rk[1],2));
  double lk = sqrt(pow(ri[0]-rj[0],2)+pow(ri[1]-rj[1],2));

  double lam1 = li*li*(lj*lj+lk*lk-li*li);
  double lam2 = lj*lj*(lk*lk+li*li-lj*lj);
  double lam3 = lk*lk*(li*li+lj*lj-lk*lk);

  double clam = lam1 + lam2 + lam3;

  double dclam_dri[3] = {-4*(li*li+lk*lk-lj*lj)*(rk[0]-ri[0])+4*(li*li+lj*lj-lk*lk)*(ri[0]-rj[0]),
                         -4*(li*li+lk*lk-lj*lj)*(rk[1]-ri[1])+4*(li*li+lj*lj-lk*lk)*(ri[1]-rj[1]),
                         -4*(li*li+lk*lk-lj*lj)*(rk[2]-ri[2])+4*(li*li+lj*lj-lk*lk)*(ri[2]-rj[2])} ;

  double dlam1_dri[3] = {2*li*li*(-(rk[0]-ri[0])+(ri[0]-rj[0])),
                         2*li*li*(-(rk[1]-ri[1])+(ri[1]-rj[1])),
                         2*li*li*(-(rk[2]-ri[2])+(ri[2]-rj[2]))} ; 

  double dlam2_dri[3] = {-2*(li*li+lk*lk-2*lj*lj)*(rk[0]-ri[0])+2*lj*lj*(ri[0]-rj[0]),
                         -2*(li*li+lk*lk-2*lj*lj)*(rk[1]-ri[1])+2*lj*lj*(ri[1]-rj[1]),
                         -2*(li*li+lk*lk-2*lj*lj)*(rk[2]-ri[2])+2*lj*lj*(ri[2]-rj[2])} ;

  double dlam3_dri[3] = {2*(li*li+lj*lj-2*lk*lk)*(ri[0]-rj[0])-2*lk*lk*(rk[0]-ri[0]),
                         2*(li*li+lj*lj-2*lk*lk)*(ri[1]-rj[1])-2*lk*lk*(rk[1]-ri[1]),
                         2*(li*li+lj*lj-2*lk*lk)*(ri[2]-rj[2])-2*lk*lk*(rk[2]-ri[2])} ;

  double d1[3] = {(clam*dlam1_dri[0]-lam1*dclam_dri[0])/(clam*clam),
                  (clam*dlam1_dri[1]-lam1*dclam_dri[1])/(clam*clam),
                  (clam*dlam1_dri[2]-lam1*dclam_dri[2])/(clam*clam)} ;

  double d2[3] = {(clam*dlam2_dri[0]-lam2*dclam_dri[0])/(clam*clam),
                  (clam*dlam2_dri[1]-lam2*dclam_dri[1])/(clam*clam),
                  (clam*dlam2_dri[2]-lam2*dclam_dri[2])/(clam*clam)} ;

  double d3[3] = {(clam*dlam3_dri[0]-lam3*dclam_dri[0])/(clam*clam),
                  (clam*dlam3_dri[1]-lam3*dclam_dri[1])/(clam*clam),
                  (clam*dlam3_dri[2]-lam3*dclam_dri[2])/(clam*clam)} ;

  Jac[0] = ri[0]*d1[0]+lam1/clam+rj[0]*d2[0]+rk[0]*d3[0];
  Jac[2] = ri[0]*d1[1]+rj[0]*d2[1]+rk[0]*d3[1];

  Jac[3] = ri[1]*d1[0]+rj[1]*d2[0]+rk[1]*d3[0];
  Jac[1] = ri[1]*d1[1]+lam1/clam+rj[1]*d2[1]+rk[1]*d3[1];

}

/*------------------------------------------------------------------------*/

int FixVolVoro::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)

{

  int i,j,m;

  m = 0;

  tagint **dtf = atom->iarray[index];
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int k = 0; k < max_faces; k++) {
        buf[m++] = ubuf(dtf[j][k]).d;
    }
  }

  return m;
  
}

void FixVolVoro::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  tagint **dtf = atom->iarray[index];
  for (i = first; i < last; i++) {
      for (int j = 0; j < max_faces; j++) {
        dtf[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }

}

/* ---------------------------------------------------------------------- */

int FixVolVoro::pack_reverse_comm(int n, int first, double *buf)
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

void FixVolVoro::unpack_reverse_comm(int n, int *list, double *buf)
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

double FixVolVoro::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax*6 * sizeof(double);
  return bytes;
}
