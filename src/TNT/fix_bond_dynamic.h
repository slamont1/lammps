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
FixStyle(bond/dynamic,FixBondDynamic);
// clang-format on
#else

#ifndef LMP_FIX_BOND_DYNAMIC_H
#define LMP_FIX_BOND_DYNAMIC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondDynamic : public Fix {
 public:
  FixBondDynamic(class LAMMPS *, int, char **);
  ~FixBondDynamic() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void post_integrate() override;
  void post_integrate_respa(int, int) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;

 protected:
  int me, nprocs;

  // Default arguments
  int nevery,iatomtype,jatomtype,btype;
  double ka,kd,cutsq;

  // Seed for random numbers
  int seed;

  // Max bond count for size of fbd
  int maxbond;

  // Flags for keywords
  int flag_bell, flag_prob, flag_rouse, flag_critical, flag_mol;

  // Explicit probabilities for flag_prob
  double prob_attach, prob_detach;

  // Force sensitivity for flag_bell
  double f0;

  // Lengthscale for flag_rouse
  double b2;

  // Critical length for flag_critical
  double r2_critical;

  // Pointers for random numbers and neighbor list
  class RanMars *random;
  class NeighList *list;

  // Flag for initializing fbd
  int countflag;

  // Maximum number of atoms
  int nmax;

  // Pointers for attachment/detachment algorithms
  tagint *partner, *finalpartner;
  double *distsq, *probabilities;
  double **partners_probs, **partners_probs_f;
  tagint **partners_possible, **partners_possible_f;
  int *npos;
  int **partners_success;

  // Character array for id of fix property/atom
  char *new_fix_id;

  // Index of fix property/atom
  int index;

  // Flag for forward communication
  int commflag;

  // For respa integration
  int nlevels_respa;

  // Internal methods/functions
  void process_broken(int, int);
  void process_created(int, int);

};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix bond/create/dynamic command

Self-explanatory.

E: Invalid bond type in fix bond/create/dynamic command

Self-explanatory.

E: Cannot use fix bond/create/dynamic with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/dynamic command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/dynamic cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/dynamic angle type is invalid

Self-explanatory.

E: Fix bond/create/dynamic dihedral type is invalid

Self-explanatory.

E: Fix bond/create/dynamic improper type is invalid

Self-explanatory.

E: Cannot yet use fix bond/create/dynamic with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/create/dynamic needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

E: New bond exceeded bonds per atom in fix bond/create/dynamic

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/dynamic

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

E: Fix bond/create/dynamic induced too many angles/dihedrals/impropers per atom

See the read_data command for info on setting the "extra angle per
atom", etc header values to allow for additional angles, etc to be
formed.

E: Special list size exceeded in fix bond/create/dynamic

See the read_data command for info on setting the "extra special per
atom" header value to allow for additional special values to be
stored.

W: Fix bond/create/dynamic is used multiple times or with fix bond/break - may not work as expected

When using fix bond/create/dynamic multiple times or in combination with
fix bond/break, the individual fix instances do not share information
about changes they made at the same time step and thus it may result
in unexpected behavior.

*/