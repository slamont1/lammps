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

#include "fix_customforce.h"

#include "arg_info.h"
#include "error.h"
#include "atom.h"
#include "atom_masks.h"
#include "compute.h"
#include "domain.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"
#include "comm.h"
#include "cell.hh"
#include "group.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <vector>
#include <bits/stdc++.h>
#include <random>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace std::chrono;
using namespace voro;

enum { NONE, CONSTANT, EQUAL, ATOM };

/* ---------------------------------------------------------------------- */

FixCustomForce::FixCustomForce(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), nvalues(0), which(nullptr), argindex(nullptr), value2index(nullptr),
    ids(nullptr), idregion(nullptr), region(nullptr), voro_data(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal fix customforce command: not sufficient args");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world, &nprocs);

  srand(10);

  dynamic_group_allow = 1;

  /*This fix takes in input as (1.) global array produced by fix vector command and (2.) per-atom array
  produced by compute voronoi*/
  
  energy_peratom_flag = 1;
  virial_peratom_flag = 1;
  thermo_energy = thermo_virial = 1;

  respa_level_support = 1;
  ilevel_respa = 0;

  nvalues = narg-3;

  // expand args if there's a wildcard  character "*" (can reset nvalues)
  // see fix_ave_atom

  int expand = 0;
  char **earg;
  nvalues = utils::expand_args(FLERR,nvalues,&arg[3],1,earg,lmp);

  if (earg != &arg[3]) expand = 1;
  arg = earg;

  /*parse values*/

  which = new int[nvalues];         //pointer to a 1D interger array
  argindex = new int[nvalues];      //"
  ids = new char*[nvalues];         // ponter to 2D character array/2D string
  value2index = new int[nvalues];   // ponter to a 2D integer array

  for(int i = 0; i < nvalues; i++){

    ids[i] = nullptr;

    ArgInfo argi(arg[i]);

    which[i] = argi.get_type();       // Will check for argument type 
    argindex[i] = argi.get_index1();
    ids[i] = argi.copy_name();        // store the user defined name of compute, eg c_voro so id[i]= voro

    if ((which[i] == ArgInfo::UNKNOWN) || (which[i] == ArgInfo::NONE)
        || (argi.get_dim() > 1))
        error->all(FLERR,"Illegal fix customforce command 1");
  }


  // if wildcard expansion occurred, free earg memory from expand_args()

  if (expand) {
    for (int i = 0; i < nvalues; i++) delete [] earg[i];
    memory->sfree(earg);
  }

 // setup and error check

 for (int i = 0; i < nvalues; i++) {
    if (which[i] == ArgInfo::COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix customfoce does not exist");
      if (modify->compute[icompute]->peratom_flag == 0)
        error->all(FLERR,
                   "Fix customforce compute does not calculate per-atom values");
      if (argindex[i] == 0 &&
          modify->compute[icompute]->size_peratom_cols != 0)
        error->all(FLERR,"Fix customforce compute does not "
                   "calculate a per-atom vector");
      if (argindex[i] && modify->compute[icompute]->size_peratom_cols == 0)
        error->all(FLERR,"Fix customforce compute does not "
                   "calculate a per-atom array");
      if (argindex[i] &&
          argindex[i] > modify->compute[icompute]->size_peratom_cols)
        error->all(FLERR,"Fix customforce compute array is accessed out-of-range");

    } else if (which[i] == ArgInfo::FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix customforce does not exist");
      if (argindex[i] == 0 && modify->fix[ifix]->vector_flag == 0)
        error->all(FLERR,
                   "Fix customforce fix does not calculate a global vector");
      if (argindex[i] && modify->fix[ifix]->array_flag == 0)
        error->all(FLERR,"Fix customforce fix does not calculate a global array");
      if (argindex[i] && argindex[i] > modify->fix[ifix]->size_array_cols)
        error->all(FLERR,"Fix ave/histo fix array is accessed out-of-range");
    }
  }
  // parse values for optional args

  /*this is not needed for now but if needed in future will require more modifications*/

  nevery = 1;  // Using default value for now

  // int iarg = nvalues+1; //////////////CHECK THIS AFTER WARDS
  // while (iarg < narg) {
  //   if (strcmp(arg[iarg], "every") == 0) {
  //     if (iarg + 2 > narg) error->all(FLERR, "Illegal fix customforce command 2");
  //     nevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
  //     if (nevery <= 0) error->all(FLERR, "Illegal fix customforce command");
  //     iarg += 2;
  //   } else if (strcmp(arg[iarg], "region") == 0) {
  //     if (iarg + 2 > narg) error->all(FLERR, "Illegal fix customforce command");
  //     region = domain->get_region_by_id(arg[iarg + 1]);
  //     if (!region) error->all(FLERR, "Region {} for fix customforce does not exist", arg[iarg + 1]);
  //     idregion = utils::strdup(arg[iarg + 1]);
  //     iarg += 2;
  //   } else
  //     error->all(FLERR, "Illegal fix customforce command");
  // }

}

/* ---------------------------------------------------------------------- */

FixCustomForce::~FixCustomForce()
{
  delete[] idregion;
  delete[] which;
  delete[] argindex;
  for (int m = 0; m < nvalues; m++) delete[] ids[m];
  delete[] ids;
  delete[] value2index;  
  
  memory->destroy(voro_data);
  
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) }

int FixCustomForce::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCustomForce::init()
{
  // set indices and check validity of all computes and variables

  // set index and check validity of region
 
  /*For future when we include nevery and region ids*/
  // if (idregion) {
  //   region = domain->get_region_by_id(idregion);
  //   if (!region) error->all(FLERR, "Region {} for fix customforce does not exist", idregion);
  // }


  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }

  for (int m = 0; m < nvalues; m++) {
    if (which[m] == ArgInfo::COMPUTE) {
      int icompute = modify->find_compute(ids[m]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/atom does not exist");
      value2index[m] = icompute;
    }
    else if (which[m] == ArgInfo::FIX) {
      int ifix = modify->find_fix(ids[m]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix custom does not exist");
      value2index[m] = ifix;
    }
    else value2index[m] = -1;
  }
}


/* ---------------------------------------------------------------------- */

void FixCustomForce::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixCustomForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS(BEGIN) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

//Function Description: get the circumcircle of a triangle

void calc_cc(double coords_array[][2], const vector<int>& v1,  double *CC){

  double x0 = coords_array[v1[0]][0];
  double x1 = coords_array[v1[1]][0];
  double x2 = coords_array[v1[2]][0];

  double y0 = coords_array[v1[0]][1];
  double y1 = coords_array[v1[1]][1];
  double y2 = coords_array[v1[2]][1];

  double a = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
  double b = sqrt(pow(x2-x0,2)+pow(y2-y0,2));
  double c = sqrt(pow(x1-x0,2)+pow(y1-y0,2));

 // Find Angles A, B and C

  double A = acos((b*b+c*c-a*a)/(2*b*c));
  double B = acos((a*a+c*c-b*b)/(2*a*c));
  double C = acos((a*a+b*b-c*c)/(2*a*b));

  CC[0] = (x0*sin(2*A)+x1*sin(2*B)+x2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // x coord of circumcenter
  CC[1] = (y0*sin(2*A)+y1*sin(2*B)+y2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // y coord of circumcenter
  CC[2] = sqrt(pow(CC[0]-x0,2)+pow(CC[1]-y0,2));  // radius of circumcircle
}

//Function decription: check if a point lies inside the circumcircle

int InsideCC(double **x, int i, const vector<double> &v){
    // find circum circle for the potential DT:

  double dist = sqrt(pow(x[i][0]-v[0],2) + pow(x[i][1]-v[1],2));
  if (dist < v[2]) return 1;   // introduce the tolerance here if you need to
  else return 0;
}

//FUnction decription: if edge is an internal/shared edge for polygon or not

int edgeshared(const vector<vector<int>> &v,int k, int kk){

int ev1,ev2;

if (kk == 0){
  ev1 = v[k][0];
  ev2 = v[k][1];
}
else if(kk == 1){
  ev1 = v[k][1];
  ev2 = v[k][2];
}
else if(kk == 2){
  ev1 = v[k][2];
  ev2 = v[k][0];
}

for (int i = 0; i < v.size(); i++){
  if (i != k){
    if (ev1 == v[i][0] && ev2 == v[i][1]) return 1;
    else if (ev1 == v[i][1] && ev2 == v[i][0]) return 1;
    else if (ev1 == v[i][1] && ev2 == v[i][2]) return 1;
    else if (ev1 == v[i][2] && ev2 == v[i][1]) return 1;
    else if (ev1 == v[i][0] && ev2 == v[i][2]) return 1;
    else if (ev1 == v[i][2] && ev2 == v[i][0]) return 1;
  }
}

return 0;
}
 
// function description: Order the vertices in CCW manner  

 void order_vertices_list (vector<int>& cvl, const vector<vector<double>> &dtcc, int cid){

 // first find the leftmost-bottommost vertex of voronoi polygon

  double ymin = INFINITY;
  int first = -1;    // initialise index for minimum y coordinate
  float epsilon = 0.0001f;
  vector<int> temp;

  for(int i = 0; i < cvl.size(); i++){
     if (ymin > dtcc[cvl[i]][1]){
      ymin = dtcc[cvl[i]][1];
      first = cvl[i];
    }
  }

  double theta[cvl.size()], theta_sorted[cvl.size()];

  // compute angles of remaining vertices w.r.t the first vertex

  for (int i = 0; i < cvl.size(); i++){
    if (cvl[i] != first){
      double x1 = dtcc[first][0];
      double y1 = dtcc[first][1];
      double x2 = dtcc[cvl[i]][0];
      double y2 = dtcc[cvl[i]][1];

      double a1 = (x2-x1)/sqrt(pow(x2-x1,2)+pow(y2-y1,2)); // x comp of vector joining both vertices
      theta[i] = acos(a1); // [0, M_PI]
    } else {
      theta[i] = -1*M_PI;
    }
    theta_sorted[i] = theta[i];
  }

  // Now sort the vertices based on their angles
  sort(theta_sorted, theta_sorted + cvl.size());

  for (int i = 0; i < cvl.size(); i++){
    for (int j = 0; j < cvl.size(); j++){
      if (theta_sorted[i] == theta[j]) temp.push_back(cvl[j]);
    }
  }

  cvl = temp;

}

// function description: get voronoi neighbors list (local ids) for an owned atom

void get_voro_neighs(int i, const vector<vector<int>> &DTmesh, const vector<int> &cvl, vector<int> &cnl){

// First traverse though the vertex list for cell i to identify the rows of corresponding DTs
  for (int j = 0; j < cvl.size(); j++){
    int id = cvl[j]; // store the index of that delaunay triangle
    for (int k = 0; k < 3; k++){
      if (DTmesh[id][k] != i && std::find(cnl.begin(), cnl.end(), DTmesh[id][k]) == cnl.end()) cnl.push_back(DTmesh[id][k]);
    }
  }
}

// helper function: finds the cross product of 2 vectors

void getCP(double *cp, double *v1, double *v2){
  cp[0] = v1[1]*v2[2]-v1[2]*v2[1];
  cp[1] = v1[2]*v2[0]-v1[0]*v2[2];
  cp[2] = v1[0]*v2[1]-v1[1]*v2[0];
}

// helper function: normalizes a vector

void normalize(double *v){
  double norm = pow(pow(v[0],2) + pow(v[1],2) + pow(v[2],2), 0.5);
  v[0] = v[0]/norm;
  v[1] = v[1]/norm;
  v[2] = v[2]/norm;
}

void Jacobian(double **x, vector<int> &dt, int p, double drmu_dri[][3]){
  // identify j and k

  int j,k;

  if (dt[0] == p) {
      j = dt[1];
      k = dt[2];
  }
  else if (dt[1] == p){
      j = dt[0];
      k = dt[2];
  }
  else if (dt[2] == p){
      j = dt[0];
      k = dt[1];
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

  drmu_dri[0][0] = ri[0]*d1[0]+lam1/clam+rj[0]*d2[0]+rk[0]*d3[0];
  drmu_dri[0][1] = ri[0]*d1[1]+rj[0]*d2[1]+rk[0]*d3[1];
  drmu_dri[0][2] = ri[0]*d1[2]+rj[0]*d2[2]+rk[0]*d3[2];

  drmu_dri[1][0] = ri[1]*d1[0]+rj[1]*d2[0]+rk[1]*d3[0];
  drmu_dri[1][1] = ri[1]*d1[1]+lam1/clam+rj[1]*d2[1]+rk[1]*d3[1];
  drmu_dri[1][2] = ri[1]*d1[2]+rj[1]*d2[2]+rk[1]*d3[2];

  drmu_dri[2][0] = ri[2]*d1[0]+rj[2]*d2[0]+rk[2]*d3[0];
  drmu_dri[2][1] = ri[2]*d1[1]+rj[2]*d2[1]+rk[2]*d3[1];
  drmu_dri[2][2] = ri[2]*d1[2]+lam1/clam+rj[2]*d2[2]+rk[2]*d3[2];

}

// helper function: multiplies a vector and a matrix

void vector_matrix(double *result, double *vec, double mat[][3]){
  for(int i = 0; i < 3; i++){
    result[i] = vec[0]*mat[0][i] + vec[1]*mat[1][i] + vec[2]*mat[2][i];
  }
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (END) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

void FixCustomForce::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int *mask = atom->mask;
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  double dt = update->dt;

  if (update->ntimestep % nevery) return;

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double *cut = comm->cutghost;

  if (update->ntimestep % nevery) return;

  // virial setup

  v_init(vflag);

  if (lmp->kokkos)
    atom->sync_modify(Host, (unsigned int) (F_MASK | MASK_MASK), (unsigned int) F_MASK);

  // update region if necessary

  if (region) region->prematch();

  int me = comm->me;               //current rank value

  // accumulate results of attributes,computes,fixes,variables to local copy
  // compute/fix/variable may invoke computes so wrap with clear/add (from fix ave atom)

  vector<double> params;
  voro_data = new double*[nall];                  
  for(int i = 0; i < nall; i++){
    voro_data[i] = new double[3];
  }

  for (int i = 0; i < nall; i++){            
    for (int m = 0; m < 3; m++){
      voro_data[i][m] = 0.0;
    }
  }

 modify->clearstep_compute();
 int n,j;

  for(int m = 0; m < nvalues; m++){
    n = value2index[m];
    j = argindex[m];

    if (which[m] == ArgInfo::COMPUTE) {
      Compute *compute = modify->compute[n];
      if (!(compute->invoked_flag & Compute::INVOKED_PERATOM)) {
        compute->compute_peratom();
        compute->invoked_flag |= Compute::INVOKED_PERATOM;
      }

      if (j == 0) {
        double *compute_vector = compute->vector_atom;
        for (int i = 0; i < nlocal; i++)                                     
          if (mask[i] & groupbit) voro_data[i][m] = compute_vector[i];
      } else {
        int jm1 = j - 1;
        double **compute_array = compute->array_atom;
        for (int i = 0; i < nlocal; i++)                                         
          if (mask[i] & groupbit) voro_data[i][jm1] = compute_array[i][jm1];
      }
  } else if (which[m] == ArgInfo::FIX) {
    Fix *fix = modify->fix[n];
    if (j == 0) {
      int num = fix->size_vector;
      for (int i = 0; i < num; i++) params.push_back(fix->compute_vector(i));
    } else {
      int num1 = fix->size_array_rows;
      int num2 = fix->size_array_cols;
      for (int i = 0; i < num2; i++) params.push_back(fix->compute_array(0,i));
    }   
  }
  }

  // forward communication of voronoi data:

  commflag = 1;
  comm->forward_comm(this,3);

  // Read the model parameters

  Elasticity = params[0]; 
  Apref = params[1];

 /*Construct Delaunay Triangulation for a set of nall points using Bowyer Watson Algorithm*/

 //declare dynamic containers to store details of Delaunay Triangulation

 vector<vector<int>> Del_Tri_mesh;
 vector<vector<double>> Del_Tri_cc; 

 /*Find min/max of x-y coords to find the super triangle*/ 

 double ymax = -1*INFINITY, ymin = INFINITY;
 double xmax = -1*INFINITY, xmin = INFINITY; 

 for (int i = 0; i < nall; i++){
  if (ymax < x[i][1]) ymax = x[i][1];
  if (ymin > x[i][1]) ymin = x[i][1];
  if (xmax < x[i][0]) xmax = x[i][0];
  if (xmin > x[i][0]) xmin = x[i][0];
 }

 double dmax;

 if (xmax-xmin > ymax-ymin) dmax = 3*(xmax-xmin);
 else dmax = 3*(ymax-ymin);

 double xcen = 0.5*(xmin+xmax);
 double ycen = 0.5*(ymin+ymax);

 // add super triangle to the Del_Tri_mesh

 double coords_array[nall+3][2] = {0.0};

 for (int i = 0; i < nall; i++){
  coords_array[i][0] = x[i][0];
  coords_array[i][1] = x[i][1];
 }

 coords_array[nall][0] = xcen-0.866*dmax;
 coords_array[nall][1] = ycen-0.5*dmax;
 coords_array[nall+1][0] = xcen+0.866*dmax;
 coords_array[nall+1][1] = ycen-0.5*dmax;
 coords_array[nall+2][0] = xcen;
 coords_array[nall+2][1] = ycen+dmax;

  double temp_CC_mod[3] = {0.0};

  Del_Tri_mesh.push_back({nall, nall+1, nall+2, 0});                      // add the super triangle which is flagged as intersecting (0)
  double temp_CC[3] = {0.0};
  calc_cc(coords_array, Del_Tri_mesh[0], temp_CC);
  Del_Tri_cc.push_back({temp_CC[0], temp_CC[1], temp_CC[2]});

 for (int i = 0; i < nall; i++){

  vector<vector<int>> bad_Del_Tri;
  vector<vector<int>> bound_edge;

  for (int j = 0; j < Del_Tri_mesh.size(); j++){
    if (InsideCC(x, i, Del_Tri_cc[j])) {
      bad_Del_Tri.push_back({Del_Tri_mesh[j][0], Del_Tri_mesh[j][1], Del_Tri_mesh[j][2]});
      Del_Tri_mesh[j][3] = 0;      // mark the triangle to be removed
    }  
    else Del_Tri_mesh[j][3] = 1;   // mark the triangle as valid for now
  }

  //Find the boundary edges for the corresponding polygon

  for (int k = 0; k < bad_Del_Tri.size(); k++){
    for (int kk = 0; kk < 3; kk++){
      if (!edgeshared(bad_Del_Tri,k, kk)){
        if (kk == 0) bound_edge.push_back({bad_Del_Tri[k][0], bad_Del_Tri[k][1]});
        else if(kk == 1) bound_edge.push_back({bad_Del_Tri[k][1], bad_Del_Tri[k][2]}); 
        else if(kk == 2) bound_edge.push_back({bad_Del_Tri[k][2], bad_Del_Tri[k][0]});
      }           
    }   
  }

  // for each triangle in bad triangle list remove from triangulation

  for (int l = 0; l < Del_Tri_mesh.size(); l++){
     if (Del_Tri_mesh[l][3] == 0){
        Del_Tri_mesh.erase(Del_Tri_mesh.begin() + l);       // remove triangle from triangulation list
        Del_Tri_cc.erase(Del_Tri_cc.begin() + l);           // Do the same for auxiliary info list
        l--; 
     }
  }

  for (int m=0; m < bound_edge.size(); m++){                           
    Del_Tri_mesh.push_back({i, bound_edge[m][0], bound_edge[m][1], 0});  
    calc_cc(coords_array, Del_Tri_mesh[Del_Tri_mesh.size()-1], temp_CC);
    Del_Tri_cc.push_back({temp_CC[0], temp_CC[1], temp_CC[2]});
   } 
 }

 // Clean up the triangulation mesh

  for (int i = 0; i < Del_Tri_mesh.size(); i++){
    if (Del_Tri_mesh[i][0] >= nall || Del_Tri_mesh[i][1] >= nall || Del_Tri_mesh[i][2] >= nall){
        Del_Tri_mesh.erase(Del_Tri_mesh.begin() + i);       // remove triangle from triangulation list
        Del_Tri_cc.erase(Del_Tri_cc.begin() + i);           // Do the same for auxiliary info list
        i--; 
    }
  }

  vector<vector<int>> cell_vertices_list(nall);

  for(int i = 0; i < nall; i++){
    for(int j = 0; j < Del_Tri_mesh.size(); j++){
      if(i == Del_Tri_mesh[j][0] || i == Del_Tri_mesh[j][1] || i == Del_Tri_mesh[j][2]){
        cell_vertices_list[i].push_back(j);         // store the vertex id in the row for cell i
      }
    }
  }

  /* Now, we want to order the vertices so that we are looping in a consistent 
      direction for all cells*/

  for(int i = 0; i < nall; i++){
    if(!cell_vertices_list[i].empty()) order_vertices_list(cell_vertices_list[i], Del_Tri_cc, i);
  }

  // create a neighbours list for [0,nlocal) i.e. owned atoms to store their voronoi neighbors

  vector<vector<int>> cell_neighs_list(nlocal);

 for (int i = 0; i < nlocal; i++){
      get_voro_neighs(i, Del_Tri_mesh, cell_vertices_list[i], cell_neighs_list[i]);
      cell_neighs_list[i].push_back(i);  // to add force contribution from the cell itself
  }

 /*Cell center based vertex model force calculation*/

  for(int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit){

    double F_t1[3] = {0.0}; //t1 stands for term 1 in the vertex model force equation which is the area term

    for(int j = 0; j < cell_neighs_list[i].size(); j++){
      int current_cell = cell_neighs_list[i][j];
      double vertex_force_sum_t1[3] = {0.0};

      // First term values needed
      double area = voro_data[current_cell][0];
      double elasticity_area = (Elasticity/2)*(area-Apref);

      int num_vert = cell_vertices_list[current_cell].size();
      int vcount = 0;
      int current_vert;

      // Looping through vertices of cell j
      while(vcount < num_vert){

        current_vert = cell_vertices_list[current_cell][vcount];

        // calculate jacobian and forces only if vertex belongs to cell i as well:
        if (std::find(cell_vertices_list[i].begin(), cell_vertices_list[i].end(), current_vert) != cell_vertices_list[i].end()){ 

          int pn[2] = {0};    // store previous and next vertex ids
          if(vcount == 0){
            pn[0] = cell_vertices_list[current_cell][num_vert - 1];
            pn[1] = cell_vertices_list[current_cell][1];
          }
          else if(vcount == num_vert - 1){
            pn[0] = cell_vertices_list[current_cell][vcount - 1];
            pn[1] = cell_vertices_list[current_cell][0];
          }
          else{
            pn[0] = cell_vertices_list[current_cell][vcount - 1];
            pn[1] = cell_vertices_list[current_cell][vcount + 1];
          }

          // First term stuff
          double rprevnext[3] = {Del_Tri_cc[pn[1]][0]-Del_Tri_cc[pn[0]][0],
                                 Del_Tri_cc[pn[1]][1]-Del_Tri_cc[pn[0]][1],
                                 0.0};
          double cp[3] = {0};
          double N[3] = {0,0,1}; // normal vector to the plane of cell layer (2D)
          getCP(cp,rprevnext,N);
          
          double drmu_dri[3][3] = {0.0};
          Jacobian(x, Del_Tri_mesh[current_vert], i, drmu_dri);                      

          double result_t1[3] = {0};
          
          // Term 1 forces
          vector_matrix(result_t1, cp, drmu_dri);
          vertex_force_sum_t1[0] += result_t1[0];
          vertex_force_sum_t1[1] += result_t1[1];
          vertex_force_sum_t1[2] += result_t1[2];
      }      
        vcount++;
      }
      F_t1[0] += elasticity_area*vertex_force_sum_t1[0];
      F_t1[1] += elasticity_area*vertex_force_sum_t1[1]; 
      F_t1[2] += elasticity_area*vertex_force_sum_t1[2];
    }
    double fx = -F_t1[0]; 
    double fy = -F_t1[1]; 
    double fz = -F_t1[2];
    f[i][0] += fx;
    f[i][1] += fy;
    f[i][2] += fz;
  }
 }
 
}

/* ---------------------------------------------------------------------- */

void FixCustomForce::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixCustomForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

int FixCustomForce::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)

{

  int i,j,k,m;

  m = 0;
 if(commflag == 1){
    for(i = 0; i < n; i++){
      j = list[i];
      for(k = 0; k < 3; k++){
        buf[m++] = voro_data[j][k];
      }
    }
  }
  return m;
}

void FixCustomForce::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,last;

  m = 0;
  last = first + n;

 if (commflag == 1) {
    for (i = first; i < last; i++){
      for(j = 0; j < 3; j++){
        voro_data[i][j] = buf[m++];
      }
    }
  }
}


