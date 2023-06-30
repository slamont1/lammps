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

#include "bond_wlc.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

#define PI 3.14159265358979323846264

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondWLC::~BondWLC()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(lp);
    memory->destroy(L);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLC::compute(int eflag, int vflag)
{
  int i1, i2, n, type;
  double delx, dely, delz, rsq, ebond, fbond;
  double lam, term1, term2;

  ebond = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx * delx + dely * dely + delz * delz;

    // Determine stretch ratio

    lam = sqrt(rsq)/L[type];

    // if lam -> 1, then chain is approaching contour length
        // issue a warning
    // if lam > 2 something serious is wrong, abort

    if (lam > 0.99) {
        error->warning(FLERR, "WLC bond too long: {} {:.8}", update->ntimestep, lam);
        if (lam > 2.0) error->one(FLERR, "Bad WLC bond");
        lam = 0.99;
    }

    // Calculate force magnitude

    term1 = (4.0*L[type]*lam)/(lp[type]*PI*pow((pow(lam,2.0)-1),2.0));
    term2 = (lp[type]*pow(PI,2.0)*lam)/L[type];
    fbond = -(term1-term2);

    // Energy

    if (eflag) {
        term1 = lp[type]*pow(PI,2.0)*(1-pow(lam,2.0))/2.0/L[type];
        term2 = 2.0*L[type]/( PI*lp[type]*(1-pow(lam,2.0)) );
        ebond = term1+term2;
    }

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx * fbond;
      f[i1][1] += dely * fbond;
      f[i1][2] += delz * fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx * fbond;
      f[i2][1] -= dely * fbond;
      f[i2][2] -= delz * fbond;
    }

    if (evflag) ev_tally(i1, i2, nlocal, newton_bond, ebond, fbond, delx, dely, delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLC::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(lp, np1, "bond:lp");
  memory->create(L, np1, "bond:L");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondWLC::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double lp_one = utils::numeric(FLERR, arg[1], false, lmp);
  double L_one = utils::numeric(FLERR, arg[2], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    lp[i] = lp_one;
    L[i] = L_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ---------------------------------------------------------------------- */

double BondWLC::equilibrium_distance(int i)
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondWLC::write_restart(FILE *fp)
{
  fwrite(&lp[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&L[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondWLC::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &lp[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &L[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&lp[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&L[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondWLC::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp, "%d %g %g\n", i, lp[i], L[i]);
}

/* ---------------------------------------------------------------------- */

double BondWLC::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
  double lam = sqrt(rsq)/L[type];

  // if lam -> 1, then chain is approaching contour length
    // issue a warning
  // if lam > 2 something serious is wrong, abort

  if (lam > 0.99) {
    error->warning(FLERR, "WLC bond too long: {} {:.8}", update->ntimestep, lam);
    if (lam > 2.0) error->one(FLERR, "Bad WLC bond");
    lam = 0.99;
  }

  double term1 = (4.0*L[type]*lam)/(lp[type]*PI*pow((pow(lam,2.0)-1),2.0));
  double term2 = (lp[type]*pow(PI,2.0)*lam)/L[type];
  fforce = -(term1-term2);

  // Energy
  term1 = lp[type]*pow(PI,2.0)*(1-pow(lam,2.0))/2.0/L[type];
  term2 = 2.0*L[type]/( PI*lp[type]*(1-pow(lam,2.0)) );
  double eng = term1+term2;

  return eng;
}

/* ---------------------------------------------------------------------- */

void *BondWLC::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "lp") == 0) return (void *) lp;
  if (strcmp(str, "L") == 0) return (void *) L;
  return nullptr;
}
