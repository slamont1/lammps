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

#include "bond_pade.h"

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

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondPADE::~BondPADE()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(b);
    memory->destroy(N);
  }
}

/* ---------------------------------------------------------------------- */

void BondPADE::compute(int eflag, int vflag)
{
  double ebond = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (int n = 0; n < nbondlist; n++) {
    int i1 = bondlist[n][0];
    int i2 = bondlist[n][1];
    int type = bondlist[n][2];

    double delx = x[i1][0] - x[i2][0];
    double dely = x[i1][1] - x[i2][1];
    double delz = x[i1][2] - x[i2][2];

    double rsq = delx * delx + dely * dely + delz * delz;
    double r = sqrt(rsq);

    // Determine stretch ratio

    double Nb = N[type] * b[type];
    double lam = r/Nb;

    // if lam -> 1, then chain is approaching contour length
        // issue a warning
    // if lam > 2 something serious is wrong, abort

    if (lam > 0.99) {
        error->warning(FLERR, "PADE bond too long: {} {:.8}", update->ntimestep, lam);
        if (lam > 2.0) error->one(FLERR, "Bad PADE bond");
        lam = 0.99;
    }

    // Calculate force magnitude

    double numer = lam*(3.0 - lam*lam);
    double denom = 1.0 - lam*lam;
    double fbond = 0.0;

    if (r > 0.0)
      fbond = -numer/denom/b[type]/r;
    else
      fbond = 0.0;

    // Energy

    if (eflag) {
        double term1 = lam*lam/2.0;
        double term2 = log(1.0 - lam*lam);
        ebond = N[type]*(term1 - term2);
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

void BondPADE::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(b, np1, "bond:b");
  memory->create(N, np1, "bond:N");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondPADE::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double b_one = utils::numeric(FLERR, arg[1], false, lmp);
  double N_one = utils::numeric(FLERR, arg[2], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    b[i] = b_one;
    N[i] = N_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ---------------------------------------------------------------------- */

double BondPADE::equilibrium_distance(int i)
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondPADE::write_restart(FILE *fp)
{
  fwrite(&b[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&N[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondPADE::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &b[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &N[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&b[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&N[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondPADE::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp, "%d %g %g\n", i, b[i], N[i]);
}

/* ---------------------------------------------------------------------- */

double BondPADE::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
  double r = sqrt(rsq);
  double Nb = N[type] * b[type];
  double lam = r/Nb;

  // if lam -> 1, then chain is approaching contour length
    // issue a warning
  // if lam > 2 something serious is wrong, abort

  if (lam > 0.99) {
    error->warning(FLERR, "PADE bond too long: {} {:.8}", update->ntimestep, lam);
    if (lam > 2.0) error->one(FLERR, "Bad PADE bond");
    lam = 0.99;
  }

  double numer = lam*(3.0 - lam*lam);
  double denom = 1.0 - lam*lam;

  if (r > 0.0)
    fforce = -numer/denom/b[type]/r;
  else
    fforce = 0.0;

  double term1 = lam*lam/2.0;
  double term2 = log(1.0 - lam*lam);
  double eng = N[type]*(term1 - term2);

  return eng;
}

/* ---------------------------------------------------------------------- */

void BondPADE::born_matrix(int type, double rsq, int /*i*/, int /*j*/, double &du, double &du2)
{
  double r = sqrt(rsq);
  double Nb = N[type] * b[type];
  double lam = r/Nb;

  du2 = 0.0;
  du = 0.0;

  // if lam -> 1, then chain is approaching contour length
    // issue a warning
  // if lam > 2 something serious is wrong, abort

  if (lam > 0.99) {
    error->warning(FLERR, "PADE bond too long: {} {:.8}", update->ntimestep, lam);
    if (lam > 2.0) error->one(FLERR, "Bad PADE bond");
    lam = 0.99;
  }

  double numer = lam*(3.0 - lam*lam);
  double denom = 1.0 - lam*lam;

  if (r > 0.0) {
    du = -numer/denom/b[type];
    
    double bb = b[type];
    double NN = N[type];
    double numer2 = 3.0*bb*bb*bb*bb*NN*NN*NN*NN+rsq*rsq;
    double denom2 = bb*bb*bb*NN*NN-bb*rsq;
    du2 = numer2/denom2/denom2/NN;
  }
}

/* ---------------------------------------------------------------------- */

void *BondPADE::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "b") == 0) return (void *) b;
  if (strcmp(str, "N") == 0) return (void *) N;
  return nullptr;
}
