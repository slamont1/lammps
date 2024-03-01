.. index:: fix bond/dynamic

fix bond/dynamic command
=======================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID bond/dynamic Nevery atomtype bondtype ka kd cutoff keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* bond/dynamic = style name of this fix command
* Nevery = attempt bond creation and deletion every this many steps
* atomtype = type of atom that can form dynamic bonds
* bondtype = type of created and destroyed bonds
* ka = forward kinetic rate of bond creation (inverse time units)
* kd = reverse kinetic rate of bond deletion (inverse time units)
* cutoff = 2 atoms separated by less than cutoff can bond (distance units)
* zero or more keyword/value pairs may be appended to args
* keyword = *maxbond* or *mol* or *prob* or *critical* or *skip*

  .. parsed-literal::

      *maxbond* values = Nbonds
         Nbonds = max # of bonds of bondtype each atom can have
      *mol* values = flag_mol
         flag_mol = 0 or 1 to consider atoms on the same molecule
      *prob* values = prob_attach, prob_detach, seed
         prob_attach = create a bond with this probability if otherwise eligible
         prob_detach = delete a bond with this probability if otherwise eligible
         seed = random number seed (positive integer)
      *critical* values = length_critical
         length_critical = critical length after which a bond permanently breaks
      *skip* values = Nevery_skip
         Nevery_skip = skip dynamic bonding on multiples of Nevery_skip
      

Examples
""""""""

.. code-block:: LAMMPS

   fix 5 all bond/dynamic 10 1 1 1.0 1.0 0.8
   fix 5 all bond/dynamic 10 1 1 2.0 0.5 1.0 prob 1.0 0.0 1234
   fix 5 all bond/dynamic 10 1 1 1.0 0.0 0.8 maxbond 5
   fix 5 all bond/dynamic 10 1 1 1.0 1.0 0.8 bell 10.0 1.0

Description
"""""""""""

Create and destroy dynamic bonds as a simulation runs according to
kinetic rates of attachment and detachment *ka* and *kd*. This can be used to model
weak (temporary) interactions in a coarse-grained molecular system.
In this context, a bond means an interaction between a pair of atoms 
computed by the :doc:`bond_style <bond_style>` command. 

A check for possible new or broken bonds is performed every *Nevery*
timesteps. A local state variable is used to keep track of eligible bonds
that can be broken on each atom. Each unique bond is only considered for
deletion exactly once. Next, a check for possible new bonds is performed.
If two atoms I,J are within a distance *cutoff* of each other, they are
labeled as a "possible" bond pair. Bonds are then created between compatible 
pairs under a probability constraint.

The methods here are different from the :doc:`fix bond/create <fix_bond_create>
and :doc:`fix bond/break <fix_bond_break> fixes for multiple reasons. First,
the potential for bond deletion is considered on a per-bond basis. This means
that one atom may have multiple bonds deleted from it on a given timestep.
In a similar manner, a single atom may have multiple bonds created on a 
given timestep. The kinetic rates *ka* and *kd* are achieved by treating
bond creation and deletion as a Poisson process; for a given desired rate
*k*, the probability *P* of a reaction occuring (either creation or deletion)
within a timespan *dt* follows the exponential decay:

.. math::

   P = 1 - exp(-k*dt)
   
In implementation, the timespan *dt* is equal to *Nevery* times the numerical
timestep.

The *maxbond* keyword can be used to limit the number of dynamic bonds that an
atom may store. This could be used to model systems in which both static and
dynamic bonds may exist. In this case, the max/bonds/per/atom must be greater
than the sum of *maxbond* and the number of static bonds on an atom.

If the *prob* keyword is used, the rates *ka* and *kd* are not considered. Instead,
each creation and deletion event is considered with probabilities *prob_attach* and
*prob_detach*, respectively.

If the *critical* keywork is used, bonds are deleted once they have reached a length of
*length_critical*. The maximum number of bonds in the atoms storing this bond will be 
decreased by one, ensuring irreversible breaking. This is akin to breaking polymer chains
by scission.

The *mol* keyword may be used to consider creating bonds between atoms on the same molecule.
Setting the flag to 1 means do not consider bond creation between atoms on the same molecule.

The *skip* keyword may be used to skip the dynamic bonding algorithms when the timestep
is on a multiple of keyword *Nevery_skip*. This can be used to distinguish bewteen
'quasi-time' timesteps and real timesteps when attempting to load in quasi-static conditions.

Any bond that is created is assigned a bond type of *bondtype*. When a bond is created, 
data structures within LAMMPS that store bond topology are updated to reflect the
creation. All of these changes typically affect pairwise interactions between
atoms that are now part of new bonds, angles, etc.

.. note::

   One data structure that is not updated when a bond breaks are
   the molecule IDs stored by each atom.  Even though two molecules
   become one molecule due to the created bond, all atoms in the new
   molecule retain their original molecule IDs.

.. note::

   To create a new bond, the internal LAMMPS data structures that
   store this information must have space for it.  When LAMMPS is
   initialized from a data file, the list of bonds is scanned and the
   maximum number of bonds per atom is tallied.  If some atom will
   acquire more bonds than this limit as this fix operates, then the
   "extra bond per atom" parameter must be set to allow for it. See the :doc:`read_data <read_data>` or
   :doc:`create_box <create_box>` command for more details.  Note that a
   data file with no atoms can be used if you wish to add non-bonded
   atoms via the :doc:`create atoms <create_atoms>` command, e.g. for a
   percolation simulation.

.. note::

   LAMMPS stores and maintains a data structure with a list of the
   first, second, and third neighbors of each atom (within the bond topology of
   the system) for use in weighting pairwise interactions for bonded
   atoms.  Note that adding a single bond always adds a new first neighbor
   but may also induce \*many\* new second and third neighbors, depending on the
   molecular topology of your system.  The "extra special per atom"
   parameter must typically be set to allow for the new maximum total
   size (first + second + third neighbors) of this per-atom list.  There are 2
   ways to do this.  See the :doc:`read_data <read_data>` or
   :doc:`create_box <create_box>` commands for details.

Note that even if your simulation starts with no bonds, you must
define a :doc:`bond_style <bond_style>` and use the
:doc:`bond_coeff <bond_coeff>` command to specify coefficients for the
*bondtype*\ .

Computationally, each timestep this fix operates, it loops over
neighbor lists and computes distances between pairs of atoms in the
list.  It also communicates between neighboring processors to
coordinate which bonds are created.  Moreover, if any bonds are
created, neighbor lists must be immediately updated on the same
timestep.  This is to insure that any pairwise interactions that
should be turned "off" due to a bond creation, because they are now
excluded by the presence of the bond and the settings of the
:doc:`special_bonds <special_bonds>` command, will be immediately
recognized.  All of these operations increase the cost of a timestep.
Thus you should be cautious about invoking this fix too frequently.

.. note::

   Creating a bond typically alters the energy of a system.  You
   should be careful not to choose bond creation criteria that induce a
   dramatic change in energy.  For example, if you define a very stiff
   harmonic bond and create it when 2 atoms are separated by a distance
   far from the equilibrium bond length, then the 2 atoms will oscillate
   dramatically when the bond is formed.  More generally, you may need to
   thermostat your system to compensate for energy changes resulting from
   created bonds (and angles, dihedrals, impropers).

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.  None of the :doc:`fix_modify <fix_modify>` options are
relevant to this fix.

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the TNT package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

Related commands
""""""""""""""""

:doc:`fix bond/break <fix_bond_break>`, :doc:`fix bond/react <fix_bond_react>`, :doc:`fix bond/swap <fix_bond_swap>`,
:doc:`dump local <dump>`, :doc:`special_bonds <special_bonds>`

Default
"""""""

The option defaults are maxbond = max/bond/per/atom and flag_mol = 0
