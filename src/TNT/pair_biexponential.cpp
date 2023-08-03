// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_biexponential.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairBiexponential::PairBiexponential(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairBiexponential::~PairBiexponential()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut1);
    memory->destroy(cut2);
    memory->destroy(a1);
    memory->destroy(a2);
    memory->destroy(kappa1);
    memory->destroy(kappa2);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairBiexponential::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,rinv,screening1,screening2,forceexp1,forceexp2,factor;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        rinv = 1.0/r;

        // First Term
        if (r <= cut1[itype][jtype]) {
          screening1 = exp(-kappa1[itype][jtype]*r);
          forceexp1 = a1[itype][jtype] * screening1;
        } else {
          screening1 = 0.0;
          forceexp1 = 0.0;
        }

        // Second Term
        if (r <= cut2[itype][jtype]) {
          screening2 = exp(-kappa2[itype][jtype]*r);
          forceexp2 = a2[itype][jtype] * screening2;
        } else {
          screening2 = 0.0;
          forceexp2 = 0.0;
        }

        fpair = factor*(forceexp1 + forceexp2) * rinv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = a1[itype][jtype]/kappa1[itype][jtype] * screening1 + a2[itype][jtype]/kappa2[itype][jtype] * screening2 - offset[itype][jtype];
          evdwl *= factor;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairBiexponential::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut1, np1, np1, "pair:cut1");
  memory->create(cut2, np1, np1, "pair:cut2");
  memory->create(a1, np1, np1, "pair:a1");
  memory->create(a2, np1, np1, "pair:a2");
  memory->create(kappa1, np1, np1, "pair:kappa1");
  memory->create(kappa2, np1, np1, "pair:kappa2");
  memory->create(offset, np1, np1, "pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBiexponential::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut2[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairBiexponential::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 8) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double a1_one = utils::numeric(FLERR, arg[2], false, lmp);
  double a2_one = utils::numeric(FLERR, arg[3], false, lmp);

  double kappa1_one = utils::numeric(FLERR, arg[4], false, lmp);
  double kappa2_one = utils::numeric(FLERR, arg[5], false, lmp);

  double cut1_one = utils::numeric(FLERR, arg[6], false, lmp);
  double cut2_one = utils::numeric(FLERR, arg[7], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      a1[i][j] = a1_one;
      a2[i][j] = a2_one;
      kappa1[i][j] = kappa1_one;
      kappa2[i][j] = kappa2_one;
      cut1[i][j] = cut1_one;
      cut2[i][j] = cut2_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBiexponential::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    a1[i][j] = mix_energy(a1[i][i],a1[j][j],1.0,1.0);
    a2[i][j] = mix_energy(a2[i][i],a2[j][j],1.0,1.0);
    cut1[i][j] = mix_distance(cut1[i][i],cut1[j][j]);
    cut2[i][j] = mix_distance(cut2[i][i],cut2[j][j]);
  }

  if (offset_flag && (kappa2[i][j] != 0.0)) {
    double screening = exp(-kappa2[i][j] * (cut2[i][j]));
    offset[i][j] = a2[i][j]/kappa2[i][j] * screening;
  } else offset[i][j] = 0.0;

  a1[j][i] = a1[i][j];
  a2[j][i] = a2[i][j];
  kappa1[j][i] = kappa1[i][j];
  kappa2[j][i] = kappa2[i][j];
  offset[j][i] = offset[i][j];

  return cut2[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBiexponential::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&a1[i][j], sizeof(double), 1, fp);
        fwrite(&a2[i][j], sizeof(double), 1, fp);
        fwrite(&kappa1[i][j], sizeof(double), 1, fp);
        fwrite(&kappa2[i][j], sizeof(double), 1, fp);
        fwrite(&cut1[i][j], sizeof(double), 1, fp);
        fwrite(&cut2[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBiexponential::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &a1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &a2[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &kappa1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &kappa2[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut1[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut2[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&a1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&a2[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&kappa1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&kappa2[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut1[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut2[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBiexponential::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBiexponential::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairBiexponential::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, a2[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairBiexponential::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) fprintf(fp, "%d %d %g %g\n", i, j, a2[i][j], cut2[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairBiexponential::single(int /*i*/, int /*j*/, int itype, int jtype,
                                 double rsq,
                                 double /*factor_coul*/, double factor_lj,
                                 double &fforce)
{
  double r,rinv,screening1,screening2,forceexp1,forceexp2,phi;

  r = sqrt(rsq);
  rinv = 1.0/r;

  // First term
  if (r <= cut1[itype][jtype]) {
    screening1 = exp(-kappa1[itype][jtype]*r);
    forceexp1 = a1[itype][jtype] * screening1;
    fforce = factor_lj*forceexp1 * rinv;
  } else {
    screening1 = 0.0;
    fforce = 0.0;
  }

  // Second term
  if (r <= cut2[itype][jtype]) {
    screening2 = exp(-kappa2[itype][jtype]*r);
    forceexp2 = a2[itype][jtype] * screening2;
    fforce += factor_lj*forceexp2 * rinv;
  } else {
    screening2 = 0.0;
  }

  phi = a1[itype][jtype]/kappa1[itype][jtype] * screening1 + a2[itype][jtype]/kappa2[itype][jtype] * screening2 - offset[itype][jtype];
  return factor_lj*phi;
}
