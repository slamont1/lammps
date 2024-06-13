// clang-format off
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

#include "fix_bond_rupture.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "respa.h"
#include "update.h"

#include <cstring>
#include <utility>
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondRupture::FixBondRupture(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal fix bond/rupture command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  btype = utils::inumeric(FLERR,arg[3],false,lmp);
  double rcrit = utils::numeric(FLERR,arg[4],false,lmp);

  dynamic_group_allow = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/rupture command");
  if (rcrit < 0.0) error->all(FLERR,"Illegal fix bond/rupture command");
  rcritsq = rcrit*rcrit;

  // Flags for optional settings
  flag_mol = 0;

  // Parse remaining arguments
  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupture command");
      flag_mol = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix bond/rupture command");
  }

  // Set forward communication size
  comm_forward = 1+atom->maxspecial;

}

/* ---------------------------------------------------------------------- */

FixBondRupture::~FixBondRupture()
{

}

/* ---------------------------------------------------------------------- */

int FixBondRupture::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

// void FixBondDynamic::post_constructor()
// {
//   new_fix_id = utils::strdup(id + std::string("RUPTURE_UPDATE_SPECIAL_BONDS"));
//   modify->add_fix(fmt::format("{} {} property/atom i2_fbd_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(maxbond)));

// }

/* ---------------------------------------------------------------------- */

void FixBondRupture::init()
{

//   if (utils::strmatch(update->integrate_style,"^respa"))
//     nlevels_respa = ((Respa *) update->integrate)->nlevels;

}

/* ---------------------------------------------------------------------- */

void FixBondRupture::post_integrate()
{

  // atom count
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // bond information
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  // bond list from neighbor class
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  // atom positions
  double **x = atom->x;

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm
  comm->forward_comm();

  // loop through neighbor bond list
  int nbreak = 0;
  for (int n = 0; n < nbondlist; n++) {

    // skip bond if already broken
    if (bondlist[n][2] <= 0) continue;

    int i1 = bondlist[n][0];
    int i2 = bondlist[n][1];
    int type = bondlist[n][2];

    // Skip bonds of wrong type
    if (type != btype) continue;

    // Ensure pair is always ordered such that r0 points in
    // a consistent direction and to ensure numerical operations
    // are identical to minimize the possibility that a bond straddling
    // an mpi grid (newton off) doesn't break on one proc but not the other
    if (atom->tag[i2] < atom->tag[i1]) {
      int itmp = i1;
      i1 = i2;
      i2 = itmp;
    }

    // Find the distance between these two atoms
    double delx = x[i1][0] - x[i2][0];
    double dely = x[i1][1] - x[i2][1];
    double delz = x[i1][2] - x[i2][2];
    domain->minimum_image(delx, dely, delz);
    double rsq = delx*delx + dely*dely + delz*delz;

    // Break bonds past the critical length
    if (rsq >= rcritsq) {

        // Process broken bond
        bondlist[n][2] = 0;
        process_broken(i1,i2);

        int i,j;
        if (i1 < i2) {
          i = i1;
          j = i2;
        } else {
          i = i2;
          j = i1;
        }

        // Sum break only once
        if (j < nlocal && i < j) {
          nbreak += 1;
        } else if (atom->tag[i] < atom->tag[j]) {
          nbreak += 1;
        }
    }
  }

  // Tally creations
  int nbreak_all;
  MPI_Allreduce(&nbreak,&nbreak_all,1,MPI_INT,MPI_SUM,world);

  // trigger reneighboring
  if (nbreak_all > 0) next_reneighbor = update->ntimestep;
  if (nbreak_all == 0) return;

  // Sum global counts
  atom->nbonds -= nbreak_all;

  // Update 1-2 neighbors 
  update_special();

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  comm->forward_comm(this);

  update_topology();

}

/* ----------------------------------------------------------------------
  Update special bond list and atom bond arrays, empty broken/created lists
------------------------------------------------------------------------- */

void FixBondRupture::update_special()
{
  int i, j, m, n1;
  tagint tagi, tagj;
  int nlocal = atom->nlocal;

  tagint *slist;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  for (auto const &it : new_broken_pairs) {
    tagi = it.first;
    tagj = it.second;
    i = atom->map(tagi);
    j = atom->map(tagj);

      if (i < 0 || j < 0) {
        error->one(FLERR,"Fix bond/rupture needs ghost atoms "
                    "from further away");
      }

    // remove i from special bond list for atom j and vice versa
    // ignore n2, n3 since 1-3, 1-4 special factors required to be 1.0
    if (i < nlocal) {
      slist = special[i];
      n1 = nspecial[i][0];
      for (m = 0; m < n1; m++)
        if (slist[m] == tagj) break;
      for (; m < n1 - 1; m++) slist[m] = slist[m + 1];
      nspecial[i][0]--;
      nspecial[i][1] = nspecial[i][2] = nspecial[i][0];
    }

    if (j < nlocal) {
      slist = special[j];
      n1 = nspecial[j][0];
      for (m = 0; m < n1; m++)
        if (slist[m] == tagi) break;
      for (; m < n1 - 1; m++) slist[m] = slist[m + 1];
      nspecial[j][0]--;
      nspecial[j][1] = nspecial[j][2] = nspecial[j][0];
    }
  }
}

/* ----------------------------------------------------------------------
  Update special lists for recently broken/created bonds
  Assumes appropriate atom/bond arrays were updated, e.g. had called
      neighbor->add_temporary_bond(i1, i2, btype);
------------------------------------------------------------------------- */

void FixBondRupture::update_topology()
{

  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;

  // In theory could communicate a list of broken bonds to neighboring processors here
  // to remove restriction that users use Newton bond off

  for (int ilist = 0; ilist < neighbor->nlist; ilist++) {
    NeighList *list = neighbor->lists[ilist];

    // Skip copied lists, will update original
    if (list->copy) continue;

    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (auto const &it : new_broken_pairs) {
      tagint tag1 = it.first;
      tagint tag2 = it.second;
      int i1 = atom->map(tag1);
      int i2 = atom->map(tag2);

      if (i1 < 0 || i2 < 0) {
        error->one(FLERR,"Fix bond/rupture needs ghost atoms "
                    "from further away");
      }

      // Loop through atoms of owned atoms i j
      if (i1 < nlocal) {
        int *jlist = firstneigh[i1];
        int jnum = numneigh[i1];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          j &= SPECIALMASK;    // Clear special bond bits
          if (tag[j] == tag2) jlist[jj] = j;
        }
      }

      if (i2 < nlocal) {
        int *jlist = firstneigh[i2];
        int jnum = numneigh[i2];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj];
          j &= SPECIALMASK;    // Clear special bond bits
          if (tag[j] == tag1) jlist[jj] = j;
        }
      }
    }
  }

  new_broken_pairs.clear();

}

/* ---------------------------------------------------------------------- */

void FixBondRupture::process_broken(int i, int j)
{

  // First add the pair to new_broken_pairs
  auto tag_pair = std::make_pair(atom->tag[i], atom->tag[j]);
  new_broken_pairs.push_back(tag_pair);

  // Manually search and remove from atom arrays
  // need to remove in case special bonds arrays rebuilt
  int nlocal = atom->nlocal;

  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;

  if (i < nlocal) {
    int n = num_bond[i];

    int done = 0;
    for (int m = 0; m < n; m++) {
      if (bond_atom[i][m] == tag[j]) {
        for (int k = m; k < n - 1; k++) {
          bond_type[i][k] = bond_type[i][k + 1];
          bond_atom[i][k] = bond_atom[i][k + 1];
        }
        num_bond[i]--;
        break;
      }
      if (done) break;
    }
  }

  if (j < nlocal) {
    int n = num_bond[j];

    int done = 0;
    for (int m = 0; m < n; m++) {
      if (bond_atom[j][m] == tag[i]) {
        for (int k = m; k < n - 1; k++) {
          bond_type[j][k] = bond_type[j][k + 1];
          bond_atom[j][k] = bond_atom[j][k + 1];
        }
        num_bond[j]--;
        break;
      }
      if (done) break;
    }
  }

}

/* ---------------------------------------------------------------------- */

int FixBondRupture::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    int ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (int k = 0; k < ns; k++) {
        buf[m++] = ubuf(special[j][k]).d;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::unpack_forward_comm(int n, int first, double *buf)
{

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    int ns = (int) ubuf(buf[m++]).i;
    nspecial[i][0] = ns;
    for (int j = 0; j < ns; j++) {
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
    
}