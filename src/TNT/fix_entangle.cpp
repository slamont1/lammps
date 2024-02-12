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

#include "fix_entangle.h"

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
#include "math_const.h"
#include "random_mars.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixEntangle::FixEntangle(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  random(nullptr), array1(nullptr)
{

  // IS THE NUMBER OF ARGUMENTS CORRECT? //
  if (narg < 2) error->all(FLERR,"Illegal fix entangle command");

  // WAS NEVERY SET CORRECTLY? //
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix entangle command");

  // SOME FLAGS... worry about later //
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  dynamic_group_allow = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  seed = 1234;
  ilevel_respa = 0;
  virial_global_flag = 1;
  virial_peratom_flag = 1;
  energy_global_flag = 1;
  peratom_flag = 1;
  size_peratom_cols = 4;


  // initialize Marsaglia RNG with processor-unique seed
  // THIS IS FOR RANDOM NUMBER GENERATION 
  random = new RanMars(lmp,seed + me);

  // INITIALIZE ANY LOCAL ARRAYS //
  array1 = nullptr;
  nmax = atom->nmax;
  FixEntangle::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  if (/*peratom_flag*/ 1) {
    init_myarray();
  }

  comm_forward = 4;

  countflag = 0;


}

/* ---------------------------------------------------------------------- */

FixEntangle::~FixEntangle()
{
  delete random;

  // delete locally stored arrays
  memory->destroy(array1);

  // DELETE CALL TO FIX PROPERTY/ATOM //
  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;

}

/* ---------------------------------------------------------------------- */

int FixEntangle::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEntangle::post_constructor()
{
  // CREATE A CALL TO FIX PROPERTY/ATOM //
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom d2_nvar_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(24)));

  // RETURN THE INDEX OF OUR LOCALLY STORED ATOM ARRAY //
  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("nvar_")+id),tmp1,tmp2);
  double **nvar = atom->darray[index];
  nmax = atom->nmax;

  // nvar IS THE POINTER TO OUR STATE VARIABLE ARRAY! //
  
  // "printcounter" is later used for printing custom messages every Nth timestep
  int printcounter = 1;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  


  // ! INITIALIZE OUR NEW STATE VARIABLE ! //
  //   !! EACH ATOM WILL STORE EXACTLY 24 VALUES !!   //
  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < 4; m++) {
      if (mask[i] & groupbit) { // This checks if the atom is in the correct group
        nvar[i][m] = 0;
      }
    }
  }
  
}

/* ---------------------------------------------------------------------- */

void FixEntangle::init()
{
  if (utils::strmatch(update->integrate_style, "^respa")) {
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;
    if (respa_level >= 0)
      ilevel_respa = MIN(respa_level, nlevels_respa - 1);
    else
      ilevel_respa = nlevels_respa - 1;
  }

  // need a full neighbor list, built at request
  //neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

void FixEntangle::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixEntangle::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet")){
    post_force(vflag);
  }else
    for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
      (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel);
      post_force_respa(vflag, ilevel, 0);
      (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel);
    }

  // Only run this in the beginning of simulation (transferring nvar data from velocities to actual peratom array)

  if (countflag) return;
  countflag = 1;

  // LOCATE THE POINTER TO OUR VARIABLE
  int tmp1, tmp2;
  double **nvar = atom->darray[index];

  // THIS WILL ONLY ACTUALLY RUN ONCE!  //
  tagint *tag = atom->tag;
  double **v = atom->v;
  int nlocal = atom->nlocal;

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  
  // nvar = [Leftside monomer count | Rightside monomer count | dangling end flag or Anchor point timer | chain tagID]
  
  for (int i = 0; i < nlocal; i++) {
    nvar[i][0] = v[i][0];
    nvar[i][1] = v[i][1];
    nvar[i][2] = v[i][2];
    nvar[i][3] = -1;    // this will be assigned in the beginning of post_force()
    printf("nvar id %d : %f %f %f %f\n\n",tag[i],nvar[i][0],nvar[i][1],nvar[i][2],nvar[i][3]);
  }
  printf("\n\n\n Initial Monomer count: %f %f %f %f\n\n\n",nvar[0][0],nvar[0][1],nvar[1][0],nvar[1][1]);

  double nlock = 720;
  int nlocked = 0;
  int c1,c2;

  // for (int i = 0; i<nlocal; i++){
  //   if (nvar[i][2]!=-1 && nvar[i][2]!=5){
  //     for (int j = 0; j < num_bond[i]; j++){
  //       if (bond_type[i][j]==2){
  //         c1 = i;
  //         c2 = atom->map(bond_atom[i][j]);
  //       }
  //     }
  //     printf("\n \n locked pair : %d %d\n \n",tag[c1],tag[c2]);
  //     nvar[c1][2] = 5;
  //     nvar[c2][2] = 5;
  //     nlocked = nlocked + 2;
  //     if (nlocked > nlock || nlocked == nlock){
  //       break;
  //     }
  //   }
  // }

  
  commflag = 1;
  comm->forward_comm(this,4);
}

/* ---------------------------------------------------------------------- */

// void FixEntangle::setup_pre_exchange()
// {

//   next_reneighbor = update->ntimestep+5000;
  
//   // HI Sam, I guess a combination of this method and the increment we define for the next_reneighbor at the end of pre_exchange() LAMMPS understands when to run pre_exchange()
// }

/* ---------------------------------------------------------------------- */
// void FixEntangle::pre_exchange()
// {
//   printf("\n\n\n\n RUNNINNNNGGGGGGGGGGG \n\n\n");


//   int i,j,i_next,i_prev;
//   double xtmp,ytmp,ztmp,delx1,dely1,delz1,delx2,dely2,delz2,rsq1,rsq2,r1,r2,Fm,Fr,fm_sqr,fbond1x,fbond1y,fbond1z,fbond2x,fbond2y,fbond2z,fbond1_squared,fbond2_squared,fmx,fmy,fmz,nu1;
//   int *ilist,*jlist,*numneigh,**firstneigh;
//   tagint *slist, tagj, tagi;
//   double fbond1,fbond2,fm;
  

//   // ATOM COUNTS
//   int nlocal = atom->nlocal;
//   int nghost = atom->nghost;
//   int nall = nlocal + nghost;

//   // OUR VARIABLE
//   int tmp1, tmp2;
//   double **nvar = atom->darray[index];

//   // BOND INFORMATION
//   int *num_bond = atom->num_bond;
//   int **bond_type = atom->bond_type;
//   tagint **bond_atom = atom->bond_atom;
  
//   double **forcearray = atom->f;

//   // SPECIAL NEIGHBORS
//   int **nspecial = atom->nspecial;
//   tagint **special = atom->special;

//   // DON'T PROCEED IF THE TIMESTEP IS NOT A MULTIPLE OF NEVERY
//   if (update->ntimestep % nevery) return;


//   // acquire updated ghost atom positions
//   // necessary b/c are calling this after integrate, but before Verlet comm
//   commflag = 1;
//   comm->forward_comm(this,4);

//   // basic atom information

  
//   double **x = atom->x;
//   double **v = atom->v;
//   tagint *tag = atom->tag;
//   int *mask = atom->mask;
//   int *type = atom->type;
//   Bond *bond = force->bond;
//   double **f = atom->f;
//   tagint *molecule = atom->molecule;
//   int **bondlist = neighbor->bondlist;
//   int nbondlist = neighbor->nbondlist;
//   int Left_previous = 0;
//   int Right_previous = 0;
//   int tag_previous = 0;
//   int n,m,k,l,s;
//   int LHS_i;
//   tagint LHS_tagi,RHS_tagi;
//   int RHS_i;
//   int endpoint;
//   int ENT_COUNT;
//   int ENT_pair;
//   int jj,kk;
//   int realnext;
//   int mm;
//   double Rzero1,Rzero2;
//   double AVGnu=0;
//   double nprev;
//   double P[6];
//   double dt = update->dt;
//   int inum,jnum;

//   if (/*peratom_flag*/ 1) {
//     init_myarray();
//   }

//   int nn = 1;

//   // here we assign random probabilities for each particle so that in case of re-entanglements it can uses them
//   // these should be kept untouched until one particle has an entanglement creation
  
//   for (i=0; i<nlocal; i++){

//     for(j=4; j<24; j++){

//       if(nvar[i][j]==0){
//         nvar[i][j] = random->uniform();
//       }

//     }

//   }

//   //accessing neighbor lists
//   inum = list->inum;
//   ilist = list->ilist;
//   numneigh = list->numneigh;
//   firstneigh = list->firstneigh;

//   double Dang_end_monomer = 0;

//   double xmid,ymid,zmid,distance;

//   double Tau;
//   int done = 0;

//   // We should loop through atoms and find dangling ends to aquire their neighborlists and find the potential segments to create entanglements with
//   for (int i = 0; i<nlocal; i++){
    
//     if(nvar[i][2]==-1){ // if a particle has nvar[][2]==-1 it is a dangling end head and we can also find it's anchorpoint through bond_atom (the dangling head is only bonded to the anchorpoint)
//       int AnchorPoint = bond_atom[i][0];    // note that this is a global tag (not a local one)
//       int Head = tag[i];                    // note that this is a global tag (not a local one)
      
//       // We are checking if the particle is at the beginning of the chain or at the end of the chain and also save it's number of monomers
//       if (nvar[atom->map(Head)][0]==0){
//         Dang_end_monomer = nvar[atom->map(Head)][1];
//       }else{
//         Dang_end_monomer = nvar[atom->map(Head)][0];
//       }
      
//       // do not allow dangling ends with monomers less than 15 to create new entanglements
//       if (Dang_end_monomer<15){
//         continue;
//       }

//       //access the neighborlist of our anchorpoint
//       jlist = firstneigh[atom->map(AnchorPoint)];
//       jnum = numneigh[atom->map(AnchorPoint)];

//       // create the array for potential list of segments (Seg_Array) and also for re-entanglement rate (Rate_array)
//       // These arrays have length equal to "jnum" which is the number of neighbors of our anchorpoint (the number of potential segments is definitely less than number of neighbors)
//       int Nseg = 0;
//       int Seg_Array[jnum][2];
//       double Rate_Array[jnum][2];

//       for (kk=0; kk<jnum; kk++){
//         Seg_Array[kk][0] = 0;
//         Seg_Array[kk][1] = 0;

//         Rate_Array[kk][0] = 0;
//         Rate_Array[kk][1] = 0;
//       }

//       // loop through neighbors of our anchorpoint to find the potential segments around it
//       for (jj=0; jj<jnum; jj++){
      
//         j=jlist[jj];
//         if (!(mask[j] & groupbit)) continue;
//         j &= NEIGHMASK;
        
//         // do not allow self-entanglements
//         if(molecule[j]==molecule[atom->map(AnchorPoint)]){
//           continue;
//         }
        
//         // second loop tries to find the particle bonded to "j" to see if there is a potential segment
//         for (int counter=0;counter<jnum;counter++){
//           int rep_flag = 0;
//           k = jlist[counter];
          
//           // if "k" & "j" are on the same molecule and connected to each other, they will form a potential segment "j-k"
//           if(molecule[k]==molecule[j] && (nvar[k][3]==nvar[j][3]-1)){
//             // We should check if this segment is already stored to avoid duplicates
//             for (nn=0; nn<Nseg; nn++){
//               if ((Seg_Array[nn][0]==tag[k] && Seg_Array[nn][1]==tag[j]) || (Seg_Array[nn][0]==tag[j] && Seg_Array[nn][1]==tag[k])){
//                 rep_flag = 1;
//                 break;
//               }
//             }

//             // store global IDs of "j" &  "k" as a potential segment if it is not already stored
//             if (rep_flag == 0){
//               for (int mit=0; mit<num_bond[k]; mit++){
//                 if (bond_atom[k][mit]==tag[j]){
//                   Seg_Array[Nseg][0]=tag[k];
//                   Seg_Array[Nseg][1]=tag[j];
//                   Nseg++;
//                 }
//               }
//             }
//           }

//           if(molecule[k]==molecule[j] && (nvar[k][3]==nvar[j][3]+1)){

//             for (nn=0; nn<Nseg; nn++){
//               if ((Seg_Array[nn][0]==tag[k] && Seg_Array[nn][1]==tag[j]) || (Seg_Array[nn][0]==tag[j] && Seg_Array[nn][1]==tag[k])){
//                 rep_flag = 1;
//                 break;
//               }
//             }

//             if (rep_flag == 0){
//               for (int mit=0; mit<num_bond[k]; mit++){
//                 if (bond_atom[k][mit]==tag[j]){
//                   Seg_Array[Nseg][0]=tag[k];
//                   Seg_Array[Nseg][1]=tag[j];
//                   Nseg++;
//                 }
//               }
//             }
//           }

//         }
//       }

//       // Here we aim to calculate the entanglement creation rate with each of the potential segments stored in "Seg_Array" based on our designed scaling law
//       // We will store these rates in "Rate_Array"
//       // The second column of "Rate_Array" stores a generated probability for that entanglement to be created which is aquired from "nvar"
//       for (int k=0;k<Nseg;k++){
//         xmid = (x[atom->map(Seg_Array[k][0])][0]+x[atom->map(Seg_Array[k][1])][0])/2;
//         ymid = (x[atom->map(Seg_Array[k][0])][1]+x[atom->map(Seg_Array[k][1])][1])/2;
//         zmid = (x[atom->map(Seg_Array[k][0])][2]+x[atom->map(Seg_Array[k][1])][2])/2;

//         distance = sqrt((xmid - x[atom->map(AnchorPoint)][0])*(xmid - x[atom->map(AnchorPoint)][0]) + (ymid - x[atom->map(AnchorPoint)][1])*(ymid - x[atom->map(AnchorPoint)][1]) + (zmid - x[atom->map(AnchorPoint)][2])*(zmid - x[atom->map(AnchorPoint)][2]));

//         if(distance>pow(Dang_end_monomer,0.6)){
//           Tau = Dang_end_monomer * pow(distance-pow(Dang_end_monomer,0.6),2) / abs(distance-Dang_end_monomer) + 20;
//         }else{
//           Tau = 20;
//         }
//         Rate_Array[k][0] = 1/Tau;
//         Rate_Array[k][1]=nvar[atom->map(AnchorPoint)][k+4];
//       }

//       // Now we will loop through the potential segments to see if any of them is probable enough to be chosen for entanglement creation
//       for (int k=0; k<Nseg; k++){
        
//         double check=0;
        
//         double lengthx = (x[atom->map(Seg_Array[k][0])][0] - x[atom->map(Seg_Array[k][1])][0]);
//         double lengthy = (x[atom->map(Seg_Array[k][0])][1] - x[atom->map(Seg_Array[k][1])][1]);
//         double lengthz = (x[atom->map(Seg_Array[k][0])][2] - x[atom->map(Seg_Array[k][1])][2]);

//         double length = sqrt(lengthx * lengthx + lengthy * lengthy + lengthz * lengthz);

//         // Just double checking if the two atoms listed in "Seg_Array" are connected
//         for (kk = 0; kk<num_bond[atom->map(Seg_Array[k][0])]; kk++){
//           if (bond_atom[atom->map(Seg_Array[k][0])][kk] == Seg_Array[k][1]){
//             check = 1;
//           }
//         }
//         // If they are not connected skip this segment
//         if (check == 0){
//           continue;
//         }

//         // Here we calculate the probability of entanglement creation based on the Rate and also the timer (nvar[][2])
//         // Prob_check is randomly generated probability threshold for that certain entanglement to be created
//         double Prob_check = Rate_Array[k][1];
//         double Prob_seg = 1 - pow(2.7182,-Rate_Array[k][0]*nvar[atom->map(AnchorPoint)][2]);

//         // If the calculated probability is larger than threshold and also the potential segment length is larger than 5 (adjustable threshold) create the entanglement
//         if (Prob_seg > Prob_check  && length > 5){
//           done = 1;
//           printf("Atom : %d Ent with segment : (%d %d) at time : %f\n",AnchorPoint,Seg_Array[k][0],Seg_Array[k][1],nvar[atom->map(AnchorPoint)][2]);

//           // reset the timer of the anchorpoint
//           nvar[atom->map(AnchorPoint)][2]=0;

//           // reset the random generated thresholds of our anchorpoint 
//           for(j=4; j<24; j++){
//               nvar[atom->map(AnchorPoint)][j] = random->uniform();
//           }
          
//           // calculating the position of the new entanglement to be created
//           double x_new_ent = (x[atom->map(Seg_Array[k][0])][0]+x[atom->map(Seg_Array[k][1])][0])/2 + random->uniform();
//           double y_new_ent = (x[atom->map(Seg_Array[k][0])][1]+x[atom->map(Seg_Array[k][1])][1])/2 - random->uniform();
//           double z_new_ent = (x[atom->map(Seg_Array[k][0])][2]+x[atom->map(Seg_Array[k][1])][2])/2;
 
//           // Since we can only move atoms and not re-assign their position vector we should find how much we should displace our Head particle to be in the right position according to position of the new entanglement
//           double displace_x = x_new_ent - x[atom->map(Head)][0];
//           double displace_y = y_new_ent - x[atom->map(Head)][1];
//           double displace_z = z_new_ent - x[atom->map(Head)][2];

//           // move the Head particle to the new position using calculated displacements
//           x[atom->map(Head)][0] += displace_x;
//           x[atom->map(Head)][1] += displace_y;
//           x[atom->map(Head)][2] += displace_z;

//           // Defining the position vector of Head of the new dangling end to be created
//           double *x_new_head = new double[3];
//           x_new_head[0] = x[atom->map(Head)][0];
//           x_new_head[1] = x[atom->map(Head)][1]-5;
//           x_new_head[2] = x[atom->map(Head)][2];

//           // Defining the position vector of the pair particle for the displaced old Head (Old Head and this particle will form the new entanglement together)
//           double *x_pair = new double[3];
//           x_pair[0]= (x[atom->map(Seg_Array[k][0])][0]+x[atom->map(Seg_Array[k][1])][0])/2;
//           x_pair[1]= (x[atom->map(Seg_Array[k][0])][1]+x[atom->map(Seg_Array[k][1])][1])/2;
//           x_pair[2]= (x[atom->map(Seg_Array[k][0])][2]+x[atom->map(Seg_Array[k][1])][2])/2;
          
//           // nlocal before the creation of new particles
//           int nlocal_previous = atom->nlocal;

//           // We should clear the map (one to one global ID - local ID relation)
//           if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
//           atom->nghost = 0;
//           atom->avec->clear_bonus();

//           // creation of the first atom (the one which is the second particle of newly created entanglement)
//           atom->avec->create_atom(1, x_pair);

//           // Not sure what next line does but since it is used in fix_deposit after particle creation we added it
//           modify->create_attribute(atom->nlocal-1);

//           // Not sure what next line does but since it is used in create_atom after particle creation we added it
//           atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

//           bigint nblocal = atom->nlocal;
//           MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
//           if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) error->all(FLERR, "Too many total atoms");
          
//           //add IDs for newly created atoms
//           //check that atom IDs are valid

//           if (atom->tag_enable) atom->tag_extend();
//           atom->tag_check();

//           atom->natoms += 1;

//           //if global map exists, reset it
//           //invoke map_init() b/c atom count has grown

//           if (atom->map_style != Atom::MAP_NONE) {
//           atom->map_init();
//           atom->map_set();
//           }
          
//           // Here we repeat all the steps like the previous atom creation
//           if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
//           atom->nghost = 0;
//           atom->avec->clear_bonus();
        

//           nlocal_previous = atom->nlocal;

//           atom->avec->create_atom(1, x_new_head);

//           modify->create_attribute(atom->nlocal-1);

//           //init per-atom fix/compute/variable values for created atoms

//           atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

//           //set new total # of atoms and error check

//           nblocal = atom->nlocal;
//           MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
//           if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) error->all(FLERR, "Too many total atoms");
          
//           //add IDs for newly created atoms
//           //check that atom IDs are valid

//           if (atom->tag_enable) atom->tag_extend();
//           atom->tag_check();

//           atom->natoms += 1;

//           //if global map exists, reset it
//           //invoke map_init() b/c atom count has grown

//           if (atom->map_style != Atom::MAP_NONE) {
//           atom->map_init();
//           atom->map_set();
//           }
//           atom->nghost = 0;
//           nlocal = atom->nlocal;
//           comm->borders();
          
//           // print global and local IDs of newly created atoms
//           printf("Created atom tags : %d & %d  local ids : (%d & %d) \n\n",tag[nlocal-1],tag[nlocal-2],nlocal-1,nlocal-2);
          
//           // print the Segment chosen for entanglement creation 
//           printf("Segment : (%d - %d) \n\n",Seg_Array[k][0],Seg_Array[k][1]);

//           // Assign the molecule IDs (nlocal-1 is the second atom we created which has molecule ID same as Head)
//           //                         (nlocal-2 is the first atom we created which has the molecule ID same as the potential segment)
//           molecule[nlocal-1] = molecule[atom->map(Head)];
//           molecule[nlocal-2] = molecule[atom->map(Seg_Array[k][0])];

//           // Here we are creating a bond to the Pair atom from each ends of the chosen segment
//           for (int jj=0; jj<num_bond[atom->map(Seg_Array[k][0])]; jj++){
//             if (bond_atom[atom->map(Seg_Array[k][0])][jj] == Seg_Array[k][1]){
//               bond_atom[atom->map(Seg_Array[k][0])][jj] = tag[nlocal-2];
//             }
//           }

//           for (int jj=0; jj<num_bond[atom->map(Seg_Array[k][1])]; jj++){
//             if (bond_atom[atom->map(Seg_Array[k][1])][jj] == Seg_Array[k][0]){
//               bond_atom[atom->map(Seg_Array[k][1])][jj] = tag[nlocal-2];
//             }
//           }

//           // We are dividing the monomers between two subchains of the chosen segment
//           if (nvar[atom->map(Seg_Array[k][0])][3]==nvar[atom->map(Seg_Array[k][1])][3]+1){
//             nvar[nlocal-2][0] = nvar[atom->map(Seg_Array[k][1])][1]/2;
//             nvar[nlocal-2][1] = nvar[atom->map(Seg_Array[k][0])][0]/2;
//             nvar[nlocal-2][2] = 0;

//             nvar[atom->map(Seg_Array[k][1])][1] = nvar[atom->map(Seg_Array[k][1])][1]/2;
//             nvar[atom->map(Seg_Array[k][0])][0] = nvar[atom->map(Seg_Array[k][0])][0]/2;
//           }

//           if (nvar[atom->map(Seg_Array[k][0])][3]==nvar[atom->map(Seg_Array[k][1])][3]-1){
//             nvar[nlocal-2][0] = nvar[atom->map(Seg_Array[k][0])][1]/2;
//             nvar[nlocal-2][1] = nvar[atom->map(Seg_Array[k][1])][0]/2;
//             nvar[nlocal-2][2] = 0;

//             nvar[atom->map(Seg_Array[k][1])][0] = nvar[atom->map(Seg_Array[k][1])][0]/2;
//             nvar[atom->map(Seg_Array[k][0])][1] = nvar[atom->map(Seg_Array[k][0])][1]/2;
//           }


//           //Here we should re-tag the chains (nvar[*][3] contains tag ID of the particles in each chain. Since new particles are added to two of the chains in the system their tag IDs must be corrected)

//           if (nvar[atom->map(Seg_Array[k][0])][3]==nvar[atom->map(Seg_Array[k][1])][3]+1){
//             nvar[nlocal-2][3] = nvar[atom->map(Seg_Array[k][0])][3];
      
//             nvar[atom->map(Seg_Array[k][0])][3] = nvar[atom->map(Seg_Array[k][0])][3] + 1;
//             int next = Seg_Array[k][0];
//             next = atom->map(next);

//             while (1==1){
//               for (int jj=0; jj<num_bond[next]; jj++){
//                 if ((nvar[atom->map(bond_atom[next][jj])][3]>nvar[next][3] || nvar[atom->map(bond_atom[next][jj])][3]==nvar[next][3]) && bond_type[next][jj]==1){
//                   next = atom->map(bond_atom[next][jj]);
//                   nvar[next][3] = nvar[next][3] + 1;
//                   break;
//                 }
//               }
//               if(nvar[next][2]==-1){
//                 break;
//               }
//             } 
//           }

//           if (nvar[atom->map(Seg_Array[k][0])][3]==nvar[atom->map(Seg_Array[k][1])][3]-1){
//             nvar[nlocal-2][3] = nvar[atom->map(Seg_Array[k][1])][3];

//             nvar[atom->map(Seg_Array[k][1])][3] = nvar[atom->map(Seg_Array[k][1])][3] + 1;
//             int next = Seg_Array[k][1];
//             next = atom->map(next);
//             while (1==1){

//               for (int jj=0; jj<num_bond[next]; jj++){
//                 if ((nvar[atom->map(bond_atom[next][jj])][3]>nvar[next][3] || nvar[atom->map(bond_atom[next][jj])][3]==nvar[next][3]) && bond_type[next][jj]==1){
//                   next = atom->map(bond_atom[next][jj]);
//                   nvar[next][3] = nvar[next][3] + 1;
//                   break;
//                 }
//               }
//               if(nvar[next][2]==-1){
//                 break;
//               }
//             } 
//           }

//           // connecting the Pair particle to the ends of the potential segment
//           bond_atom[nlocal-2][0] = Seg_Array[k][0];
//           bond_type[nlocal-2][0] = 1;
//           num_bond[nlocal-2]++;

//           bond_atom[nlocal-2][1] = Seg_Array[k][1];
//           bond_type[nlocal-2][1] = 1;
//           num_bond[nlocal-2]++;

//           // connecting The Pair particle to the old Head (to make them an entanglement junction pair)
//           bond_atom[nlocal-2][2] = Head;
//           bond_type[nlocal-2][2] = 2;
//           num_bond[nlocal-2]++;

//           // reset the state variable for the old Head
//           if (nvar[atom->map(Head)][3]!=1){
//             nvar[atom->map(Head)][2] = 0;
//             nvar[atom->map(Head)][0] = Dang_end_monomer * 0.8;
//             nvar[atom->map(Head)][1] = Dang_end_monomer * 0.2;

//             // we should also change some of the state variable of the connected atom to our old Head
//             int prev = bond_atom[atom->map(Head)][0];
//             nvar[atom->map(prev)][1] = Dang_end_monomer * 0.8;;
//             nvar[nlocal-1][3] = nvar[atom->map(Head)][3] + 1;
//             nvar[nlocal-1][2] = -1;
//             nvar[nlocal-1][0] = Dang_end_monomer * 0.2;
//             nvar[nlocal-1][1] = 0;

//             // Here we save the updated state variable to re-check them in the beginning of next timestep
//             for (int ii=0; ii<nlocal; ii++){
//               printf("atom %d with local id %d : %f  %f  %f  %f\n",tag[ii],ii,nvar[ii][0],nvar[ii][1],nvar[ii][2],nvar[ii][3]);
//               dummy[ii][0] = tag[ii];
//               dummy[ii][1] = nvar[ii][0];
//               dummy[ii][2] = nvar[ii][1];
//               dummy[ii][3] = nvar[ii][2];
//               dummy[ii][4] = nvar[ii][3];
//               dummy_flag = -1;
//             }

//           }else{ // in this case in addition to the update of state variable all the atoms in the chain should have their tag IDs updated
//             nvar[atom->map(Head)][2] = 0;
//             nvar[atom->map(Head)][1] = Dang_end_monomer * 0.8;
//             nvar[atom->map(Head)][0] = Dang_end_monomer * 0.2;
  
//             int nextt = bond_atom[atom->map(Head)][0];
//             nvar[atom->map(nextt)][0] = Dang_end_monomer * 0.8;

//             nvar[nlocal-1][2] = -1;
//             nvar[nlocal-1][1] = Dang_end_monomer * 0.2;
//             nvar[nlocal-1][0] = 0;
//             nvar[nlocal-1][3] = 1;
//             nvar[atom->map(Head)][3] = 2;


//             // Shifting the tag IDs
//             int next = atom->map(Head);

//             while (1==1){
//               for (int jj=0; jj<num_bond[next]; jj++){
//                 if((nvar[atom->map(bond_atom[next][jj])][3]>nvar[next][3] || nvar[atom->map(bond_atom[next][jj])][3]==nvar[next][3]) && (bond_type[next][jj]==1)){
//                   next = atom->map(bond_atom[next][jj]);
//                   nvar[next][3] = nvar[next][3] + 1;
//                   break;
//                 }
//               }
//               if(nvar[next][2]==-1){
//                 break;
//               }
//             }

//             // Here we save the updated state variable to re-check them in the beginning of next timestep
//             for (int ii=0; ii<nlocal; ii++){
//               printf("atom %d with local id %d : %f  %f  %f  %f\n",tag[ii],ii,nvar[ii][0],nvar[ii][1],nvar[ii][2],nvar[ii][3]);
//               dummy[ii][0] = tag[ii];
//               dummy[ii][1] = nvar[ii][0];
//               dummy[ii][2] = nvar[ii][1];
//               dummy[ii][3] = nvar[ii][2];
//               dummy[ii][4] = nvar[ii][3];
//               dummy_flag = -1;
//             }

//           }



//           // Creating the entanglement pair by connecting old Head to the newly created Pair particle
//           bond_atom[atom->map(Head)][1] = tag[nlocal-2];
//           bond_type[atom->map(Head)][1] = 2;
//           num_bond[atom->map(Head)]++;

//           // connecting the old head to the new head
//           bond_atom[atom->map(Head)][2] = tag[nlocal-1];
//           bond_type[atom->map(Head)][2] = 1;
//           num_bond[atom->map(Head)]++;

//           // connecting the new head to the old head
//           bond_atom[nlocal-1][0] = Head;
//           bond_type[nlocal-1][0] = 1;
//           num_bond[nlocal-1]++;
          

//           atom->data_fix_compute_variable(nlocal_previous, atom->nlocal);

//           //set new total # of atoms and error check

//           nblocal = atom->nlocal;
//           MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
//           if (atom->natoms < 0 || atom->natoms >= MAXBIGINT) error->all(FLERR, "Too many total atoms");
          
//           //add IDs for newly created atoms
//           //check that atom IDs are valid

//           if (atom->tag_enable) atom->tag_extend();
//           atom->tag_check();

//           //if global map exists, reset it
//           //invoke map_init() b/c atom count has grown

//           if (atom->map_style != Atom::MAP_NONE) {
//           atom->map_init();
//           atom->map_set();
//           }
          
//           // breaks since the Anchorpoint has already chosen a segment and the entanglement is created
//           break;
//         }
//       }

//     }

//     // Only allows one entanglement created per timestep (2 new particles)
//     if (done==1){
//       break;
//     }

//   }

  
//   next_reneighbor += 5000;

// }
/* ---------------------------------------------------------------------- */

void FixEntangle::post_force(int vflag)
{
 
  // DEFINE ALL OF YOUR TEMPORARY VARIABLES HERE //
  int i,j,i_next,i_prev;
  double xtmp,ytmp,ztmp,delx1,dely1,delz1,delx2,dely2,delz2,rsq1,rsq2,r1,r2,Fm,Fr,fm_sqr,fbond1x,fbond1y,fbond1z,fbond2x,fbond2y,fbond2z,fbond1_squared,fbond2_squared,fmx,fmy,fmz,nu1;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist, tagj, tagi;
  double fbond1,fbond2,fm;
  

  // ATOM COUNTS
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // per-atom flow rates stored in nu[]
  double *nu = new double[nlocal];


  // these are used later for visualization and ovito purposes
  double *N_rest1 = new double[nlocal];
  double *N_rest2 = new double[nlocal];
  double *N1_0 = new double[nlocal];
  double *N2_0 = new double[nlocal];

  // Our state variable
  int tmp1, tmp2;
  double **nvar = atom->darray[index];

  // BOND INFORMATION
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  

  double **forcearray = atom->f;

  // SPECIAL NEIGHBORS
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // DON'T PROCEED IF THE TIMESTEP IS NOT A MULTIPLE OF NEVERY
  if (update->ntimestep % nevery) return;
  v_init(vflag);

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm
  // commflag = 1;
  // comm->forward_comm(this,4);

  // basic atom information
  double **x = atom->x;
  double **v = atom->v;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  Bond *bond = force->bond;
  double **f = atom->f;
  tagint *molecule = atom->molecule;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int Left_previous = 0;
  int Right_previous = 0;
  int tag_previous = 0;
  int n,m,k,l,s;
  int LHS_i;
  tagint LHS_tagi,RHS_tagi;
  int RHS_i;
  int endpoint;
  int ENT_COUNT;
  int ENT_pair;
  int jj,kk;
  int realnext;
  int mm;
  double Rzero1,Rzero2;
  double AVGnu=0;
  double nprev;
  double P[6];
  double S[6];
  double dt = update->dt;

  for (i=1 ; i<nlocal; i++){
  if (printcounter == 1){
      N1_0[i] = nvar[i][0]/N_rest1[i];
      N2_0[i] = nvar[i][1]/N_rest2[i];
  }
  }
  int inum,jnum;
        
  if (peratom_flag) {
    init_myarray();
  }


  // Here we are tagging the atoms in each chain so that we can have a sequential direction in each chain for future calculations

  for (i=0; i < nlocal; i++) {
    int RHS_tagi = -1;
    int LHS_tagi = -1;

    if(nvar[i][3]!=-1) continue; //do not proceed if this atom is already tagged

    //Here we aim to find the RHS and LHS entanglements indices to start sweeping all across the chain to count the number of entanglements in the chain of interest
    for (m=0; m < num_bond[i]; m++){
      if (bond_type[i][m]==1){
        if (LHS_tagi==-1){
          LHS_tagi=bond_atom[i][m];
        }
          else{
            RHS_tagi=bond_atom[i][m];
          }
        }
      }

    Left_previous = i;
    Right_previous = i;
    LHS_i = atom->map(LHS_tagi);
    RHS_i = atom->map(RHS_tagi);

    //sweeping leftside to count
    for (n = 0; n < nlocal; n++){
      if (nvar[LHS_i][2]==-1) {
        break;
        }    //break if the entanglement has a dangling end

      for (m=0; m < num_bond[LHS_i]; m++){
        if (bond_type[LHS_i][m]==1 & atom->map(bond_atom[LHS_i][m])!=Left_previous){
            Left_previous = LHS_i;
            LHS_i=atom->map(bond_atom[LHS_i][m]);
            break;
            
        }
      }
    }
    endpoint = LHS_i;

    //check if Right hand side atom is a dangling end to avoid extra loop
    if(RHS_tagi==-1) {
      k=-1;
      RHS_i=i;
    }
    else{
      //sweeping rightside to count
        for (k = 0; k < nlocal; k++){
          if (nvar[RHS_i][2]==-1) break;    //break if the entanglement has a dangling end
          //printf("*\n");
          for (m=0; m < num_bond[RHS_i]; m++){
            if (bond_type[RHS_i][m]==1 & atom->map(bond_atom[RHS_i][m])!=Right_previous){
                Right_previous = RHS_i;
                RHS_i=atom->map(bond_atom[RHS_i][m]);
                break;
            }
          }
        }
    }
    

    ENT_COUNT = (n + 1) + (k + 1) + 1;

    //Now we Assign tagIDs by having number of entanglement on the chain of interest
    tag_previous = endpoint;

    //check whether this entanglement is the beginning one or the ending one
    if(nvar[endpoint][0]==20){
    for (l = 0; l < ENT_COUNT; l++){
      nvar[endpoint][3]=ENT_COUNT-l;
      for (s=0; s < num_bond[endpoint]; s++){
        if (bond_type[endpoint][s]==1 & atom->map(bond_atom[endpoint][s])!=tag_previous){
            tag_previous = endpoint;
            endpoint=atom->map(bond_atom[endpoint][s]);
            break;
        }
      }
    }
    }else{
    for (l = 0; l < ENT_COUNT; l++){
      nvar[endpoint][3]=l+1;
      for (s=0; s < num_bond[endpoint]; s++){
        if (bond_type[endpoint][s]==1 & atom->map(bond_atom[endpoint][s])!=tag_previous){
            tag_previous = endpoint;
            endpoint=atom->map(bond_atom[endpoint][s]);
            break;
        }
      }
    }
    }
 
  }

  //this re-assigns the state variable in case of any re-entanglement of disentanglement (since nvar might be messed up)
  if (dummy_flag==-1){
    for (i=0; i<nlocal; i++){
      for (j=0; j<nlocal; j++){
        if (tag[i]==dummy[j][0]){
          nvar[i][0]=dummy[j][1];
          nvar[i][1]=dummy[j][2];
          nvar[i][2]=dummy[j][3];
          nvar[i][3]=dummy[j][4];
          dummy_flag = 0;
        }
      }
    }
  }

  //loop over all entanglement points to calculate forces and then the monomer sliding rates
  for (i = 0; i < nlocal; i++) { 

    // Check if the atom is in the correct group... keep for generality
    if (!(mask[i] & groupbit)) continue;
    
    // components of left-hand side chain vector
    delx1 = 0;
    dely1 = 0;
    delz1 = 0;

    // components of right-hand side chain vector
    delx2 = 0;
    dely2 = 0;
    delz2 = 0;

    // components of left-hand side force
    fbond1x = 0;
    fbond1y = 0;
    fbond1z = 0;

    // components of right-hand side force
    fbond2x = 0;
    fbond2y = 0;
    fbond2z = 0;

    // force magnitudes
    fbond1 = 0;
    fbond2 = 0;

    // second to last and second entanglements flag used later
    int secondflag = 0;
    int secondtolastflag = 0;
    

    //We need a loop to go over columns of bond_atom because each atom is connected to three/one particles to find the left-hand side atom and right-hand side atom to aquire their vectorial distances.
    //We have used the tagID here to find the previous and next atoms (stored in nvar[*][3] for each atom)

    for (int jj=0; jj < num_bond[i]; jj++){
      if (bond_type[i][jj]==1 && nvar[atom->map(bond_atom[i][jj])][3]==nvar[i][3]-1){
        delx1 = x[atom->map(bond_atom[i][jj])][0] - x[i][0];
        dely1 = x[atom->map(bond_atom[i][jj])][1] - x[i][1];
        delz1 = x[atom->map(bond_atom[i][jj])][2] - x[i][2];

        if(nvar[atom->map(bond_atom[i][jj])][2]==-1){
          secondflag = 1;
          // here if atom is the second atom in a chain it means it is an anchorpoint for a dangling end and we turn on the timer for it's re-entanglement timer at nvar[*][2]
          if (nvar[i][2] == 0){
            //nvar[i][2] = dt;
          }
        }
      }

      if (bond_type[i][jj]==1 && nvar[atom->map(bond_atom[i][jj])][3]==nvar[i][3]+1){
        delx2 = x[atom->map(bond_atom[i][jj])][0] - x[i][0];
        dely2 = x[atom->map(bond_atom[i][jj])][1] - x[i][1];
        delz2 = x[atom->map(bond_atom[i][jj])][2] - x[i][2];

        if(nvar[atom->map(bond_atom[i][jj])][2]==-1){
          secondtolastflag = 1;
          // here if atom is the second to last atom in a chain it means it is an anchorpoint for a dangling end and we turn on the timer for it's re-entanglement timer at nvar[*][2]
          if (nvar[i][2] == 0){
            //nvar[i][2] = dt;
          }
        }
      }
    }

    //calculate the periodic distance magnitude
    domain->minimum_image(delx1, dely1, delz1);
    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1=sqrt(rsq1);

    //calculate the number of monomers for a sub-chain of length r1 to be at equilibrium
    N_rest1[i] = pow(r1,1.667);
    

    domain->minimum_image(delx2, dely2, delz2);
    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2=sqrt(rsq2); 
    N_rest2[i] = pow(r2,1.667);


    // calculate the Rzero1 and Rzero2 as equilibrium length of each sides based on their number of monomers
    double Ncube1 = nvar[i][0]*nvar[i][0]*nvar[i][0];
    double Ncube2 = nvar[i][1]*nvar[i][1]*nvar[i][1];
    Rzero1 = pow(Ncube1,0.2);
    Rzero2 = pow(Ncube2,0.2);


    // here we calculate the force components from the left and right hand side sub-chains 

    if(r1!=0){ // if length of a sub-chain is zero it means that particle is the head of a dangling end
      if (r1>Rzero1 && secondflag!=1){ // if atom is the second atom in the chain the force of dangling end side is always zero
        fbond1x = (3 * 1) * (delx1 / nvar[i][0]) - 3 * (nvar[i][0]*nvar[i][0])/(r1*r1*r1*r1) * delx1/r1;       //(k * T) / (b * b) = 1
        fbond1y = (3 * 1) * (dely1 / nvar[i][0]) - 3 * (nvar[i][0]*nvar[i][0])/(r1*r1*r1*r1) * dely1/r1;       //(k * T) / (b * b) = 1
        fbond1z = (3 * 1) * (delz1 / nvar[i][0]) - 3 * (nvar[i][0]*nvar[i][0])/(r1*r1*r1*r1) * delz1/r1;  
      }
    }


    if(r2!=0){ // if length of a sub-chain is zero it means that particle is the head of a dangling end
      if (r2>Rzero2 && secondtolastflag!=1){ // if atom is the second atom in the chain the force of dangling end side is always zero
        fbond2x = (3 * 1) * (delx2 / nvar[i][1]) - 3 * (nvar[i][1]*nvar[i][1])/(r2*r2*r2*r2) * delx2/r2;       //(k * T) / (b * b) = 1
        fbond2y = (3 * 1) * (dely2 / nvar[i][1]) - 3 * (nvar[i][1]*nvar[i][1])/(r2*r2*r2*r2) * dely2/r2;       //(k * T) / (b * b) = 1
        fbond2z = (3 * 1) * (delz2 / nvar[i][1]) - 3 * (nvar[i][1]*nvar[i][1])/(r2*r2*r2*r2) * delz2/r2;
      }
    }

    //Used to print custom stuff
    if ((printcounter % 10000)==0){
      //printf("id:%d   f1 = %f , %f   f2 = %f , %f  rzeros : %f %f   r1:%f r2:%f\n",i+1,fbond1x,fbond1y,fbond2x,fbond2y,Rzero1,Rzero2,r1,r2);
      //printf("id:%d      nvar: [%f %f %f %f]\n",i+1,nvar[i][0],nvar[i][1],nvar[i][2],nvar[i][3]);
      // ...
    }
    
    // sliding friction coefficient
    double zeta = 0.01;


    // here we pass the sum of the forces on each particle to the f[] array for use of other fixes (except when particle is a dangling end head)
    // if the particle is a dangling end head we will integrate it's motion manually here and it also doesn't contribute to virial stress
    // we also calculate the virial contributions and pass them via vtally()
    if(nvar[i][2]!=-1){
      f[i][0] += (fbond1x + fbond2x);
      f[i][1] += (fbond1y + fbond2y);
      f[i][2] += (fbond1z + fbond2z);

      S[0] += (abs(delx1*fbond1x) + abs(delx2*fbond2x));
      //P[0] += sqrt(fbond1x * fbond1x + fbond1y * fbond1y) * 50 * 50 * 4;
      S[1] += (abs(dely1*fbond1y) + abs(dely2*fbond2y));
      //P[1] += sqrt(fbond1x * fbond1x + fbond1y * fbond1y) * 50 * 50 * 4;
      S[2] += (abs(delz1*fbond1z) + abs(delz2*fbond2z));
      S[3] += (abs(delx1*fbond1y) + abs(delx2*fbond2y));
      S[4] += (abs(delx1*fbond1z) + abs(delx2*fbond2z));
      S[5] += (abs(dely1*fbond1z) + abs(dely2*fbond2z));


    if (evflag) {
      P[0] += (abs(delx1*fbond1x) + abs(delx2*fbond2x));
      //P[0] += sqrt(fbond1x * fbond1x + fbond1y * fbond1y) * 50 * 50 * 4;
      P[1] += (abs(dely1*fbond1y) + abs(dely2*fbond2y));
      //P[1] += sqrt(fbond1x * fbond1x + fbond1y * fbond1y) * 50 * 50 * 4;
      P[2] += (abs(delz1*fbond1z) + abs(delz2*fbond2z));
      P[3] += (abs(delx1*fbond1y) + abs(delx2*fbond2y));
      P[4] += (abs(delx1*fbond1z) + abs(delx2*fbond2z));
      P[5] += (abs(dely1*fbond1z) + abs(dely2*fbond2z));

      v_tally(i, P);
    }

    }else{

      f[i][0] = 0;
      f[i][1] = 0;
      f[i][2] = 0;

      x[i][0] += dt * ((fbond1x + fbond2x))/(0.01);
      x[i][1] += dt * ((fbond1y + fbond2y))/(0.01);
      x[i][2] += dt * ((fbond1z + fbond2z))/(0.01);

      v[i][0] = (fbond1x + fbond2x)/(0.01);
      v[i][1] = (fbond1y + fbond2y)/(0.01);
      v[i][2] = (fbond1z + fbond2z)/(0.01);

    }

    // calculate the force magnitude from each side of the entanglement for calculation of sliding rate
    fbond1_squared = (fbond1x * fbond1x) + (fbond1y * fbond1y) + (fbond1z * fbond1z);
    fbond1 = sqrt(fbond1_squared);
    if (delx1*fbond1x<0){
      fbond1 = -fbond1;
    }

    fbond2_squared = (fbond2x * fbond2x) + (fbond2y * fbond2y) + (fbond2z * fbond2z);
    fbond2 = sqrt(fbond2_squared);
    if (delx2*fbond2x<0){
      fbond2 = -fbond2;
    }

    Fm = fbond2-fbond1;
    Fr = random->gaussian(0,2*zeta);

    // difference of tension from two sides is used to calculate monomer sliding rate at that entanglement
    nu[i] = (Fm)/zeta;  
    
    // Average sliding magnitude (sometimes printed as a measure of relaxation)
    AVGnu = AVGnu + (sqrt(nu[i]*nu[i]))/nlocal; 


  }
    if ((update->ntimestep % 100)==0){
      printf("%ld %f  %f  %f\n",update->ntimestep,S[0],S[1],S[2]);
      //printf("%f\n",AVGnu);
    }
  // forward communication of the velocities so ghost atoms store their values 
  commflag = 1;
  comm->forward_comm(this,4);
  
  /* THIS PART IS WRITTEN BY SAM I DON'T KNOW IF WE NEED IT OR NOT

    // Loop over ghost atoms, find corresponding entries of velocity and update
    // needed for ghost atoms directly bonded to owned atoms??
    for (j = nlocal; j < nall; j++) {
      // STUFF ???? MAYBE REQUIRED NOT SURE YET
    }
  */

  // Here we run thr second loop to integrate the monomer counts after one timestep of sliding
  int DIS_flag = 0;

  for (i = 0; i < nlocal; i++) {
    if (nvar[i][2]!=-1 && nvar[i][2]!=5){ //dangling ends head does not have any sliding

      // Check if the atom is in the correct group... keep for generality
      if (!(mask[i] & groupbit)) continue;
      
      for (jj=0; jj < num_bond[i]; jj++){
        if (bond_type[i][jj]==1 & nvar[atom->map(bond_atom[i][jj])][3]==nvar[i][3]-1){
          nvar[atom->map(bond_atom[i][jj])][1] = nvar[atom->map(bond_atom[i][jj])][1] - nu[i]*(dt);
        }

        if (bond_type[i][jj]==1 & nvar[atom->map(bond_atom[i][jj])][3]==nvar[i][3]+1){
          nvar[atom->map(bond_atom[i][jj])][0] = nvar[atom->map(bond_atom[i][jj])][0] + nu[i]*(dt);
        }
      }

      // turns on disentanglement flag for that to happen in the next time step at pre_exchange
      nvar[i][0] = nvar[i][0] - nu[i]*(dt);
      if (nvar[i][0]<5 && nvar[i][0]>0.001){
        DIS_flag = 1;
        nvar[i][2]=5;
      } 

      nvar[i][1] = nvar[i][1] + nu[i]*(dt);
      if (nvar[i][1]<5 && nvar[i][1]>0.001){
        DIS_flag = 1;
        nvar[i][2]=5;
      } 
    }

    // array_atom is a per-atom array which can be dumped for visualization purposes in ovito
    if(1/*peratom_flag*/){
      if(N_rest1[i]!=0){  
      array_atom[i][0] = (nvar[i][0]/N_rest1[i] - N1_0[i]) * 1/(1-N1_0[i]);
      }
      if(N_rest2[i]!=0){
      array_atom[i][1] = (nvar[i][1]/N_rest2[i] - N2_0[i]) * 1/(1-N2_0[i]);
      }  
      // array_atom[i][0] = nvar[i][0];
      // array_atom[i][1] = nvar[i][1];
      array_atom[i][2] = nvar[i][2];
      array_atom[i][3] = nvar[i][3];
    }
  }
  
  // here we update the timer of anchorpoints for re-entanglement purposes (Once at each timestep)
  // for (i=0; i<nlocal; i++){
  //   if (nvar[i][2]==-1){
  //     int anchorpoint = atom->map(bond_atom[i][0]);
  //     nvar[anchorpoint][2] += update->dt;
  //   }
  // } 

  commflag = 1;
  comm->forward_comm(this,4);

  // DISETANTNELMENT SECTION !!!!

  int NextAtom,next,prev; 
  AtomVec *avec = atom->avec;
  i=0;
if (DIS_flag == 1){
while (i<nlocal){
int begayi = 0;
  if (nvar[i][2]==-1){
    if (nvar[i][1]<1 && nvar[i][1]>0.001){
      //printf("1111111       number of atoms: %d\n",nlocal);
        //printf("%d",i+1);
      NextAtom = atom->map(bond_atom[i][0]);
      nvar[NextAtom][0]=0;
      nvar[NextAtom][2]=-1;
      n=num_bond[NextAtom];
      nvar[NextAtom][3]=nvar[NextAtom][3]-1;

      jj=0;
      while (jj < num_bond[NextAtom]){


        if(atom->map(bond_atom[NextAtom][jj])==i || bond_type[NextAtom][jj]==2){

          if (bond_type[NextAtom][jj]==2){

            ENT_pair = atom->map(bond_atom[NextAtom][jj]);
            double monomer = nvar[ENT_pair][0] + nvar[ENT_pair][1];

            bond_type[NextAtom][jj] = bond_type[NextAtom][num_bond[NextAtom]-1];
            bond_atom[NextAtom][jj] = bond_atom[NextAtom][num_bond[NextAtom]-1];
            num_bond[NextAtom]--;

            kk = 0;
            
            while(kk<num_bond[ENT_pair]){

              if (bond_type[ENT_pair][kk]==2){
                bond_type[ENT_pair][kk] = bond_type[ENT_pair][num_bond[ENT_pair]-1];
                bond_atom[ENT_pair][kk] = bond_atom[ENT_pair][num_bond[ENT_pair]-1];
                num_bond[ENT_pair]--;
                //printf("kooniENT PAIR  %d\n\n",ENT_pair+1);
                //printf("nvar 17 :%f %f %f %f \n",nvar[17][0],nvar[17][1],nvar[17][2],nvar[17][3]);
              
              } else if(bond_type[ENT_pair][kk] == 1 && nvar[atom->map(bond_atom[ENT_pair][kk])][3]>nvar[ENT_pair][3]){
                //printf("NANAT JENDast\n");
                nvar[atom->map(bond_atom[ENT_pair][kk])][0] = monomer;
                next = atom->map(bond_atom[ENT_pair][kk]);
                nvar[next][3] = nvar[next][3] - 1; 
                realnext = next;
                kk++;
                //printf("nvar 17 :%f %f %f %f \n",nvar[17][0],nvar[17][1],nvar[17][2],nvar[17][3]);
              //retagging the pair chain
              
              if (nvar[next][2]!=-1){
              while(1==1){
                //printf("KIR 333\n");
                for (int nn=0; nn<num_bond[next]; nn++){
                  if (nvar[atom->map(bond_atom[next][nn])][3]>nvar[next][3] && bond_type[next][nn]==1){
                    next = atom->map(bond_atom[next][nn]);
                    break;
                  }   
                }
                nvar[next][3]=nvar[next][3]-1;
                if(nvar[next][2]==-1){
                  break;
                }
              }
              }   
              } else if(bond_type[ENT_pair][kk] == 1 && nvar[atom->map(bond_atom[ENT_pair][kk])][3]<nvar[ENT_pair][3]){
                //printf("nvar 17 :%f %f %f %f \n",nvar[17][0],nvar[17][1],nvar[17][2],nvar[17][3]);              
                //printf("ENT PAIR  %d\n",ENT_pair+1);
                nvar[atom->map(bond_atom[ENT_pair][kk])][1] = monomer;
                prev = atom->map(bond_atom[ENT_pair][kk]);
                kk++;
              }
            }


              // should also delete bonds to the ENT_pair in next and prev atoms !!!!!!!!!!!!!!!!!!!!!!!!!!!!
              for (int mm=0; mm<num_bond[realnext]; mm++){
                if (atom->map(bond_atom[realnext][mm])==ENT_pair){
                  bond_type[realnext][mm] = bond_type[realnext][num_bond[realnext]-1];
                  bond_atom[realnext][mm] = bond_atom[realnext][num_bond[realnext]-1];
                  num_bond[realnext]--;
                  break;
                }
              }

              for (int mm=0; mm<num_bond[prev]; mm++){
                if (atom->map(bond_atom[prev][mm])==ENT_pair){
                  bond_type[prev][mm] = bond_type[prev][num_bond[prev]-1];
                  bond_atom[prev][mm] = bond_atom[prev][num_bond[prev]-1];
                  num_bond[prev]--;
                  break;
                }
              }

              bond_type[prev][num_bond[prev]] = 1;
              bond_atom[prev][num_bond[prev]] = tag[realnext];
              num_bond[prev]++;

              bond_type[realnext][num_bond[realnext]] = 1;
              bond_atom[realnext][num_bond[realnext]] = tag[prev];
              num_bond[realnext]++;              
              //printf("\n\nDEEELLLEEEEEETTTEEEDD PARTICLE : %d     local : %d\n\n",tag[ENT_pair],ENT_pair);
              if(i==nlocal-1){
                begayi = 1;
              }
              avec->copy(nlocal - 1, ENT_pair, 1);
              nlocal--;
              atom->nlocal = nlocal;   


          }else if(atom->map(bond_atom[NextAtom][jj])==i){
          bond_type[NextAtom][jj] = bond_type[NextAtom][num_bond[NextAtom]-1];
          bond_atom[NextAtom][jj] = bond_atom[NextAtom][num_bond[NextAtom]-1];
          num_bond[NextAtom]--;
          
          //printf("KIRRE MAN TOO IN ZENDEGI");
          }
        }else{
          jj++;
        }
      }
      //printf("\n\nDEEELLLEEEEEETTTEEEDD PARTICLE : %d    local : %d\n\n",tag[i],i);


      if(begayi!=1){
        avec->copy(nlocal - 1, i, 1);
      }else{
        avec->copy(nlocal-1, ENT_pair, 1);
      }
      nlocal--;
      atom->nlocal = nlocal;
      atom->natoms -= 2;
                            for(int kiri=0; kiri<nlocal; kiri++){
                  //printf("nvar %d : %f %f %f %f molecule : %d\n",tag[kiri],nvar[kiri][0],nvar[kiri][1],nvar[kiri][2],nvar[kiri][3],molecule[kiri]);
                  //printf("id : %d  (local id : %d) numbond: %d bonds to : %d %d %d\n\n",tag[kiri],kiri,num_bond[kiri],bond_atom[kiri][0],bond_atom[kiri][1],bond_atom[kiri][2]);
                }
      //printf("numbond 2 & 4:%d %d",num_bond[1],num_bond[3]);
      if (atom->map_style != Atom::MAP_NONE) {
        atom->nghost = 0;
        atom->map_init();
        atom->map_set();
      }
      //retagging the chain//
      
      while(1==1){
        //printf("\n\n\n\nNEXT ATOM : %d  (local id %d) \n\n\n\n",tag[NextAtom],NextAtom);
        for (jj=0; jj<num_bond[NextAtom]; jj++){
          if (nvar[atom->map(bond_atom[NextAtom][jj])][3]>nvar[NextAtom][3] && bond_type[NextAtom][jj]==1){
            NextAtom = atom->map(bond_atom[NextAtom][jj]);
            break;
          }
        }
        nvar[NextAtom][3]=nvar[NextAtom][3]-1;
        if(nvar[NextAtom][2]==-1){
          break;
        }
      }

for (int bipp=0; bipp<nlocal; bipp++){
  //printf("NVAR %d  (local id : %d) = %f  %f  %f  %f\n\n",tag[bipp],bipp,nvar[bipp][0],nvar[bipp][1],nvar[bipp][2],nvar[bipp][3]);
}
      
    } else if (nvar[i][0]<1 && nvar[i][0]>0.001){
      //printf("2222222222    number of atoms:%d\n",nlocal);
      NextAtom = atom->map(bond_atom[i][0]);
      nvar[NextAtom][1]=0;
      nvar[NextAtom][2]=-1;
      n=num_bond[NextAtom];
      jj=0;
      while (jj < num_bond[NextAtom]){
        if(atom->map(bond_atom[NextAtom][jj])==i || bond_type[NextAtom][jj]==2){
          //printf("Next atom : %d\n",NextAtom);
          if (bond_type[NextAtom][jj]==2){
            ENT_pair = atom->map(bond_atom[NextAtom][jj]);
            double monomer = nvar[ENT_pair][0] + nvar[ENT_pair][1];
            
            bond_type[NextAtom][jj] = bond_type[NextAtom][num_bond[NextAtom]-1];
            bond_atom[NextAtom][jj] = bond_atom[NextAtom][num_bond[NextAtom]-1];
            num_bond[NextAtom]--;
            
            kk=0;
            while(kk<num_bond[ENT_pair]){
              if (bond_type[ENT_pair][kk]==2){
                bond_type[ENT_pair][kk] = bond_type[ENT_pair][num_bond[ENT_pair]-1];
                bond_atom[ENT_pair][kk] = bond_atom[ENT_pair][num_bond[ENT_pair]-1];
                num_bond[ENT_pair]--;
              }else if(bond_type[ENT_pair][kk] == 1 && nvar[atom->map(bond_atom[ENT_pair][kk])][3]>nvar[ENT_pair][3]){
                
                nvar[atom->map(bond_atom[ENT_pair][kk])][0] = monomer;
                next = atom->map(bond_atom[ENT_pair][kk]);
                nvar[next][3] = nvar[next][3] - 1; 
                realnext = next;
                kk++;
                //printf("Number of atoms:%d\n\n",nlocal);
              //printf("nvar 10:%f %f %f %f \n",nvar[10][0],nvar[10][1],nvar[10][2],nvar[10][3]);
              //printf("nvar 13 :%f %f %f %f \n",nvar[13][0],nvar[13][1],nvar[13][2],nvar[13][3]);
                if(nvar[next][2]!=-1){
                while(1==1){
                  for (int nn=0; nn<num_bond[next]; nn++){
                    if (nvar[atom->map(bond_atom[next][nn])][3]>nvar[next][3] && bond_type[next][nn]==1){
                      next = atom->map(bond_atom[next][nn]);
                      break;
                    }   
                  }
                  nvar[next][3]=nvar[next][3]-1;
                  if(nvar[next][2]==-1){
                    break;
                  }
                }
                }   
                }else if(bond_type[ENT_pair][kk] == 1 && nvar[atom->map(bond_atom[ENT_pair][kk])][3]<nvar[ENT_pair][3]){
                nvar[atom->map(bond_atom[ENT_pair][kk])][1] = monomer;
                prev = atom->map(bond_atom[ENT_pair][kk]);
                kk++;
              }

            }

              for (int mm=0; mm<num_bond[realnext]; mm++){
                if (atom->map(bond_atom[realnext][mm])==ENT_pair){
                  bond_type[realnext][mm] = bond_type[realnext][num_bond[realnext]-1];
                  bond_atom[realnext][mm] = bond_atom[realnext][num_bond[realnext]-1];
                  num_bond[realnext]--;
                }
              }

              for (int mm=0; mm<num_bond[prev]; mm++){
                if (atom->map(bond_atom[prev][mm])==ENT_pair){
                  bond_type[prev][mm] = bond_type[prev][num_bond[prev]-1];
                  bond_atom[prev][mm] = bond_atom[prev][num_bond[prev]-1];
                  num_bond[prev]--;
                }
              }

              bond_type[prev][num_bond[prev]] = 1;
              bond_atom[prev][num_bond[prev]] = tag[realnext];
              num_bond[prev]++;

              bond_type[realnext][num_bond[realnext]] = 1;
              bond_atom[realnext][num_bond[realnext]] = tag[prev];
              num_bond[realnext]++;              
//printf("\n\nDEEELLLEEEEEETTTEEEDD PARTICLE : %d   local : %d\n\n",tag[ENT_pair],ENT_pair);
              
              if (i==nlocal-1){
                begayi = 1 ;
              }
              avec->copy(nlocal - 1, ENT_pair, 1);
              nlocal--;
              atom->nlocal = nlocal;


          }else if(atom->map(bond_atom[NextAtom][jj])==i){
          bond_type[NextAtom][jj] = bond_type[NextAtom][num_bond[NextAtom]-1];
          bond_atom[NextAtom][jj] = bond_atom[NextAtom][num_bond[NextAtom]-1];
          num_bond[NextAtom]--;
          }
        }else{
          jj++;
        }
      }
      //atom->num_bond[i] = n;
//printf("\n\nDEEELLLEEEEEETTTEEEDD PARTICLE : %d   local : %d\n\n",tag[i],i);
      if(begayi!=1){
        avec->copy(nlocal - 1, i, 1);
      }else{
        avec->copy(nlocal-1, ENT_pair, 1);
      }
      nlocal--;
      atom->nlocal = nlocal;
      atom->natoms -= 2;
                for(int kiri=0; kiri<nlocal; kiri++){
                  //printf("nvar %d : %f %f %f %f molecule : %d\n",tag[kiri],nvar[kiri][0],nvar[kiri][1],nvar[kiri][2],nvar[kiri][3],molecule[kiri]);
                  //printf("id : %d  (local id : %d) numbond: %d bonds to : %d %d %d\n\n",tag[kiri],kiri,num_bond[kiri],bond_atom[kiri][0],bond_atom[kiri][1],bond_atom[kiri][2]);
                }
    }else{
      i++;
    }
      if (atom->map_style != Atom::MAP_NONE) {
        atom->nghost = 0;
        atom->map_init();
        atom->map_set();
      }
      comm->borders();
        bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  }else{
    i++;
  }
}



} 
//printf("\n\nIN POST FORCE %d\n\n",printcounter);
commflag = 2;
comm->forward_comm(this,atom->bond_per_atom);

bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  bigint nbonds = 0;

    for (int i = 0; i < nlocal; i++) {
      if (num_bond) nbonds += num_bond[i];
    }

  if (atom->avec->bonds_allow) {
    MPI_Allreduce(&nbonds, &atom->nbonds, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (!force->newton_bond) atom->nbonds /= 2;
  }




  // for printing custom stuff
  printcounter = printcounter + 1;
  if ((printcounter % 100000)==0){
    for (int kos=0; kos<nlocal; kos++){
      if(dummy_flag==-1){
        //printf("atom %d with local id %d : %f  %f  %f  %f %f\n",tag[kos],kos,dummy[kos][0],dummy[kos][1],dummy[kos][2],dummy[kos][3],dummy[kos][4]);
      }
      //printf("atom %d with local id %d : %f  %f  %f  %f\n",tag[kos],kos,nvar[kos][0],nvar[kos][1],nvar[kos][2],nvar[kos][3]);
    }
  }

  // Forward communication of nvar! //
  commflag = 1;
  comm->forward_comm(this,4);
  
}

/* ---------------------------------------------------------------------- */

// !!!!!!!!!!!!!! IGNORE !!!!!!!!!!! //
void FixEntangle::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEntangle::min_post_force(int vflag)
{
  post_force(vflag);
}


/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixEntangle::grow_arrays(int nmax)
{
  if (1/*peratom_flag*/) {
    memory->grow(array_atom,nmax,size_peratom_cols,"fix_entangle:array_atom");
  }
}
/* ---------------------------------------------------------------------- */

void FixEntangle::init_myarray()
{
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    for (int m = 0; m < size_peratom_cols; m++) {
      array_atom[i][m] = 0.0;
    }
  }
}

/* --------------------------------------------------------------------
 copy values within local atom-based arrays
----------------------------------------------------------------- */

void FixEntangle::copy_arrays(int i, int j, int delflag)
{
  if (/*peratom_flag*/ 1) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[j][m] = array_atom[i][m];
  }
}
/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixEntangle::set_arrays(int i)
{
  if (/*peratom_flag*/ 1) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[i][m] = 0;
  }
}
/* ---------------------------------------------------------------------- */

// THIS IS WHERE WE DEFINE COMMUNICATION //
int FixEntangle::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  m = 0;

  // CAN SET MULTIPLE COMMUNICATION TYPES USING commflag //
  if (commflag == 1) {
      int tmp1, tmp2;
      index = atom->find_custom(utils::strdup(std::string("nvar_") + id),tmp1,tmp2);
      double **nvar = atom->darray[index];

      for (i = 0; i < n; i++) {
        j = list[i];
        for (k = 0; k < 4; k++) {
          buf[m++] = nvar[j][k];
        }
      }
      return m;
  }

  if (commflag == 2) {
      int *num_bond = atom->num_bond;
      int **bond_type = atom->bond_type;
      tagint **bond_atom = atom->bond_atom;

      for (i = 0; i < n; i++) {
        j = list[i];
        ns = num_bond[j];
        buf[m++] = ubuf(ns).d;
        for (k = 0; k < ns; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
        }
      }
      return m;
  }

  // EXAMPLE OF COMMUNICATION OF SPECIAL LIST BY DEFAULT //
  // int **nspecial = atom->nspecial;
  // tagint **special = atom->special;

  // m = 0;
  // for (i = 0; i < n; i++) {
  //   j = list[i];
  //   ns = nspecial[j][0];
  //   buf[m++] = ubuf(ns).d;
  //   for (k = 0; k < ns; k++)
  //     buf[m++] = ubuf(special[j][k]).d;
  // }
  // return m;
}

/* ---------------------------------------------------------------------- */

// THIS IS THE OPPOSITE OF THE PREVIOUS SCRIPT //
// ALWAYS NEED TO DEFINE BOTH PACKING AND UNPACKING //
void FixEntangle::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  // if (commflag == 1) {
    int tmp1, tmp2;
    index = atom->find_custom(utils::strdup(std::string("nvar_") + id),tmp1,tmp2);
    double **nvar = atom->darray[index];

    for (i = first; i < last; i++) {
        for (j = 0; j < 4; j++) {
          nvar[i][j] = buf[m++];
        }
    }

  // } else {
    // int **nspecial = atom->nspecial;
    // tagint **special = atom->special;

    // m = 0;
    // last = first + n;
    // for (i = first; i < last; i++) {
    //   ns = (int) ubuf(buf[m++]).i;
    //   nspecial[i][0] = ns;
    //   for (j = 0; j < ns; j++)
    //     special[i][j] = (tagint) ubuf(buf[m++]).i;
    // }
  // }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixEntangle::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(double);
  if (peratom_flag) bytes += (double)nmax*size_peratom_cols*sizeof(double);
  return bytes;
}
