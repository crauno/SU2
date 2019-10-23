/*!
 * \file numerics_direct_mesh.cpp
 * \brief This file contains the routines for setting the mesh pseudo-elastic problem.
 * \author R. Sanchez
 * \version 6.2.0 "Falcon"
 *
 * The current SU2 release has been coordinated by the
 * SU2 International Developers Society <www.su2devsociety.org>
 * with selected contributions from the open-source community.
 *
 * The main research teams contributing to the current release are:
 *  - Prof. Juan J. Alonso's group at Stanford University.
 *  - Prof. Piero Colonna's group at Delft University of Technology.
 *  - Prof. Nicolas R. Gauger's group at Kaiserslautern University of Technology.
 *  - Prof. Alberto Guardone's group at Polytechnic University of Milan.
 *  - Prof. Rafael Palacios' group at Imperial College London.
 *  - Prof. Vincent Terrapon's group at the University of Liege.
 *  - Prof. Edwin van der Weide's group at the University of Twente.
 *  - Lab. of New Concepts in Aeronautics at Tech. Institute of Aeronautics.
 *
 * Copyright 2012-2019, Francisco D. Palacios, Thomas D. Economon,
 *                      Tim Albring, and the SU2 contributors.
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../include/numerics_structure.hpp"
#include <limits>

CFEAMeshElasticity::CFEAMeshElasticity(unsigned short val_nDim, unsigned short val_nVar, unsigned long val_nElem, CConfig *config) : CFEALinearElasticity() {

  DV_Val         = NULL;
  FAux_Dead_Load = NULL;
  Rho_s_i        = NULL;
  Rho_s_DL_i     = NULL;
  Nu_i           = NULL;

  nDim = val_nDim;
  nVar = val_nVar;

  unsigned long iVar;

  E = config->GetDeform_ElasticityMod();
  Nu = config->GetDeform_PoissonRatio();
  Compute_Lame_Parameters();

  switch (config->GetDeform_Stiffness_Type()) {
  case INVERSE_VOLUME:
  case SOLID_WALL_DISTANCE:
    element_based = true;
    Nu = config->GetDeform_Coeff();
    break;
  case CONSTANT_STIFFNESS:
    element_based = false;
    break;
  }

  E_i  = NULL;
  if (element_based){
    E_i         = new su2double[val_nElem];
    for (iVar = 0; iVar < val_nElem; iVar++){
      E_i[iVar] = E;
    }
  }

  KAux_ab = new su2double* [nDim];
  for (iVar = 0; iVar < nDim; iVar++) {
    KAux_ab[iVar] = new su2double[nDim];
  }

  if (nDim == 2) {
    Ba_Mat = new su2double* [3];
    Bb_Mat = new su2double* [3];
    D_Mat  = new su2double* [3];
    Ni_Vec  = new su2double [4];            /*--- As of now, 4 is the maximum number of nodes for 2D problems ---*/
    GradNi_Ref_Mat = new su2double* [4];
    GradNi_Curr_Mat = new su2double* [4];
    for (iVar = 0; iVar < 3; iVar++) {
      Ba_Mat[iVar]      = new su2double[nDim];
      Bb_Mat[iVar]      = new su2double[nDim];
      D_Mat[iVar]       = new su2double[3];
    }
    for (iVar = 0; iVar < 4; iVar++) {
      GradNi_Ref_Mat[iVar]   = new su2double[nDim];
      GradNi_Curr_Mat[iVar]   = new su2double[nDim];
    }
  }
  else if (nDim == 3) {
    Ba_Mat = new su2double* [6];
    Bb_Mat = new su2double* [6];
    D_Mat  = new su2double* [6];
    Ni_Vec  = new su2double [8];           /*--- As of now, 8 is the maximum number of nodes for 3D problems ---*/
    GradNi_Ref_Mat = new su2double* [8];
    GradNi_Curr_Mat = new su2double* [8];
    for (iVar = 0; iVar < 6; iVar++) {
      Ba_Mat[iVar]      = new su2double[nDim];
      Bb_Mat[iVar]      = new su2double[nDim];
      D_Mat[iVar]        = new su2double[6];
    }
    for (iVar = 0; iVar < 8; iVar++) {
      GradNi_Ref_Mat[iVar]   = new su2double[nDim];
      GradNi_Curr_Mat[iVar]   = new su2double[nDim];
    }
  }

}

CFEAMeshElasticity::~CFEAMeshElasticity(void) {

}

