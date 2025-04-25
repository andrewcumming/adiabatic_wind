//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"

void HydroInnerX1(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void HydroOuterX1(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


Real KK, gam, gm1, rho0;

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief isothermal wind
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  gam = peos->GetGamma();
  gm1 = gam - 1.0;
  KK = pin->GetReal("problem","K");
  rho0 = pin->GetReal("problem","rho0");
  //Real r2 = 1.0/(1.0 - gam*KK/gm1);
  Real r2 = 1.1;
  std::cout << "r2=" << r2 << "\n";
    
  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        Real dens = 1e-10;
        if (r <= r2) {
            dens = rho0 * std::pow( 1.0 + (gm1/gam/KK)*(1.0/r-1.0), 1.0/gm1);
        }       
        phydro->u(IDN,k,j,i) = dens;
        phydro->u(IEN,k,j,i) = KK*std::pow(dens,gam)/gm1;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }  
  return;
}


void Mesh::InitUserMeshData(ParameterInput *pin) {
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  return;
}


void HydroOuterX1(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real re = pco->x1v(ie);
          Real r = pco->x1v(ie+i);          
          prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie) * re*re / (r*r);
          prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
          prim(IVY,k,j,ie+i) = 0.0;
          prim(IVZ,k,j,ie+i) = 0.0;
          prim(IPR,k,j,ie+i) = prim(IPR,k,j,ie);
        }
      }
    }
}


void HydroInnerX1(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{  
    
  for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          Real r = pco->x1v(is-i);
          Real dens = rho0 * std::pow( 1.0 + (gm1/gam/KK)*(1.0/r-1.0), 1.0/gm1);
          prim(IDN,k,j,is-i) = dens;
          prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
          prim(IVY,k,j,is-i) = 0.0;
          prim(IVZ,k,j,is-i) = 0.0;
          prim(IPR,k,j,is-i) = KK*std::pow(dens,gam);
        }
    }
  }
}
