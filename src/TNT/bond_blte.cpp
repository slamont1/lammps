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

#include "bond_blte.h"

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

BondBLTE::~BondBLTE()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
	memory->destroy(L);
    memory->destroy(lp);
    memory->destroy(T);	
  }
}

/* ---------------------------------------------------------------------- */

void BondBLTE::compute(int eflag, int vflag)
{
  int i1, i2, n, type;
  double delx, dely, delz, ebond, fbond;
  double rsq, lam, r, Lmax, l_p;
  double kb, kappa_b, Temp, pi;
  
  pi = 3.141592653589793;

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

	// Max rod elongation 
	Lmax 	= L[type]; 
	// Persistence Length 
	l_p 	= lp[type];
	// Temperature 
	Temp 	= T[type];
	// Boltzmann constant 
	kb = 1;
	// End-to-end distance
	r = sqrt(rsq);
	// Non-dimensional Filament Extension Factor
	lam = r/Lmax;

    // if lam -> 1, filament streched at L
    // issue a warning and reset lam
    // if lam > 2 something serious is wrong, abort

    if (lam >= 0.99) {
      error->warning(FLERR, "Blundell-Terentjev bond too long: {} {}", update->ntimestep,  sqrt(rsq));
      if (lam > 1) error->one(FLERR, "Bad Blundell-Terentjev bond");
    }

    // Calculate Force
	
	// fbond = k_1*epsilon*exp(k_2*epsilon*epsilon)*(epsilon + 1);
	
	fbond = (-1) * (kb * Temp) * ((-1*pi*pi*l_p/(Lmax*Lmax))*lam + 4/(pi*l_p)*lam*pow(1-lam*lam, -2)) * (Lmax/r);
	
    // energy

    if (eflag) {
      ebond = (kb * Temp) * (0.5*pi*pi*l_p/Lmax*(1-lam*lam) + (2*Lmax/l_p)*(1/pi)*1/(1-lam*lam));
		//ebond = (kb*Temp)/lp) * (0.5*lam*lam + 0.25*pow(1-lam, -1) - 0.25*lam) * Lmax;
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

void BondBLTE::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(T, np1, "bond:T");
  memory->create(lp, np1, "bond:lp");
  memory->create(L, np1, "bond:L");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondBLTE::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);
  
  double T_one = utils::numeric(FLERR, arg[1], false, lmp);
  double lp_one = utils::numeric(FLERR, arg[2], false, lmp);
  double L_one = utils::numeric(FLERR, arg[3], false, lmp);
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    T[i] = T_one;
	lp[i] = lp_one;
	L[i] = L_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check if special_bond settings are valid
------------------------------------------------------------------------- */

void BondBLTE::init_style()
{
  // special bonds should be 0 1 1

  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 || force->special_lj[3] != 1.0) {
    if (comm->me == 0) error->warning(FLERR, "Use special bonds = 0,1,1 with bond style fene");
  }
}

/* ---------------------------------------------------------------------- */

double BondBLTE::equilibrium_distance(int i)
{
  return 0.0;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondBLTE::write_restart(FILE *fp)
{
  fwrite(&T[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&lp[2], sizeof(double), atom->nbondtypes, fp);
  fwrite(&L[3], sizeof(double), atom->nbondtypes, fp);  
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondBLTE::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &T[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
	utils::sfread(FLERR, &lp[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
	utils::sfread(FLERR, &L[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&T[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&lp[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&L[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondBLTE::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, T[i], lp[i], L[i]);
}

/* ---------------------------------------------------------------------- */

double BondBLTE::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
	double pi = 3.141592653589793;
	// Max rod elongation 
	double Lmax 	= L[type]; 
	// Persistence length
	double l_p = lp[type];
	// Temperature 
	double Temp 	= T[type];
	// Boltzmann constant 
	double kb = 1;
	// End-to-end distance
	double r = sqrt(rsq);
	// Non-dimensional Filament Extension Factor
	double lam = r/Lmax;

    if (lam >= 0.99) {
      error->warning(FLERR, "Blundell-Terentjev bond too long: {} {}", update->ntimestep,  sqrt(rsq));
      if (lam > 1) error->one(FLERR, "Bad Blundell-Terentjev bond");
    }

    // Calculate Force
	fforce = (-1) * (kb * Temp) * ((-1*pi*pi*l_p/(Lmax*Lmax))*lam + 4/(pi*l_p)*lam*pow(1-lam*lam, -2)) * (Lmax/r);

    // energy
	double eng = (kb * Temp) * (0.5*pi*pi*l_p/Lmax*(1-lam*lam) + (2*Lmax/l_p)*(1/pi)*1/(1-lam*lam));

  return eng;
}

/* ---------------------------------------------------------------------- */

void *BondBLTE::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "T") == 0) return (void *) T;
  if (strcmp(str, "lp") == 0) return (void *) lp;
  if (strcmp(str, "L") == 0) return (void *) L;
  return nullptr;
}
