﻿/*!
 * \file nemo_turb_diffusion.cpp
 * \brief Implementation of numerics classes to compute viscous
 *        fluxes in turbulence problems.
 * \author F. Palacios, T. Economon
 * \version 7.0.6 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
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

#include "../../../include/numerics/turbulent/nemo_turb_diffusion.hpp"

CNEMOAvgGrad_Scalar::CNEMOAvgGrad_Scalar(unsigned short val_nDim, unsigned short val_nVar,
                                         unsigned short val_nPrimVar,
                                         unsigned short val_nPrimVarGrad,
                                         bool correct_grad, const CConfig* config):
                                         CNEMONumerics(val_nDim, val_nVar, val_nPrimVar,
                                                       val_nPrimVarGrad, config),
  correct_gradient(correct_grad),
  implicit(config->GetKind_TimeIntScheme_Turb() == EULER_IMPLICIT),
  incompressible(config->GetKind_Regime() == INCOMPRESSIBLE)
{
  Proj_Mean_GradTurbVar_Normal = new su2double [nVar] ();
  Proj_Mean_GradTurbVar_Edge = new su2double [nVar] ();
  Proj_Mean_GradTurbVar = new su2double [nVar] ();

  Flux = new su2double [nVar] ();
  Jacobian_i = new su2double* [nVar];
  Jacobian_j = new su2double* [nVar];
  for (unsigned short iVar = 0; iVar < nVar; iVar++) {
    Jacobian_i[iVar] = new su2double [nVar] ();
    Jacobian_j[iVar] = new su2double [nVar] ();
  }
}

CNEMOAvgGrad_Scalar::~CNEMOAvgGrad_Scalar(void) {

  delete [] Proj_Mean_GradTurbVar_Normal;
  delete [] Proj_Mean_GradTurbVar_Edge;
  delete [] Proj_Mean_GradTurbVar;

  delete [] Flux;
  if (Jacobian_i != nullptr) {
    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      delete [] Jacobian_i[iVar];
      delete [] Jacobian_j[iVar];
    }
    delete [] Jacobian_i;
    delete [] Jacobian_j;
  }
}

CNumerics::ResidualType<> CNEMOAvgGrad_Scalar::ComputeResidual(const CConfig* config) {

  unsigned short iVar, iDim;

  AD::StartPreacc();
  AD::SetPreaccIn(Coord_i, nDim); AD::SetPreaccIn(Coord_j, nDim);
  AD::SetPreaccIn(Normal, nDim);
  AD::SetPreaccIn(TurbVar_Grad_i, nVar, nDim);
  AD::SetPreaccIn(TurbVar_Grad_j, nVar, nDim);
  if (correct_gradient) {
    AD::SetPreaccIn(TurbVar_i, nVar); AD::SetPreaccIn(TurbVar_j ,nVar);
  }
  ExtraADPreaccIn();

  //TODO ndim+7 update
  AD::SetPreaccIn(V_i, nDim+7); AD::SetPreaccIn(V_j, nDim+7);

  Density_i           = V_i[RHO_INDEX];       Density_j           = V_j[RHO_INDEX];
  Laminar_Viscosity_i = V_i[LAM_VISC_INDEX];  Laminar_Viscosity_j = V_j[LAM_VISC_INDEX];
  Eddy_Viscosity_i    = V_i[EDDY_VISC_INDEX]; Eddy_Viscosity_j    = V_j[EDDY_VISC_INDEX];

  /*--- Compute vector going from iPoint to jPoint ---*/

  dist_ij_2 = 0; proj_vector_ij = 0;
  for (iDim = 0; iDim < nDim; iDim++) {
    Edge_Vector[iDim] = Coord_j[iDim]-Coord_i[iDim];
    dist_ij_2 += Edge_Vector[iDim]*Edge_Vector[iDim];
    proj_vector_ij += Edge_Vector[iDim]*Normal[iDim];
  }
  if (dist_ij_2 == 0.0) proj_vector_ij = 0.0;
  else proj_vector_ij = proj_vector_ij/dist_ij_2;

  /*--- Mean gradient approximation ---*/
  for (iVar = 0; iVar < nVar; iVar++) {
    Proj_Mean_GradTurbVar_Normal[iVar] = 0.0;
    Proj_Mean_GradTurbVar_Edge[iVar] = 0.0;
    for (iDim = 0; iDim < nDim; iDim++) {
      su2double Mean_GradTurbVar = 0.5*(TurbVar_Grad_i[iVar][iDim] +
                                        TurbVar_Grad_j[iVar][iDim]);

      Proj_Mean_GradTurbVar_Normal[iVar] += Mean_GradTurbVar * Normal[iDim];

      if (correct_gradient)
        Proj_Mean_GradTurbVar_Edge[iVar] += Mean_GradTurbVar * Edge_Vector[iDim];
    }
    Proj_Mean_GradTurbVar[iVar] = Proj_Mean_GradTurbVar_Normal[iVar];
    if (correct_gradient) {
      Proj_Mean_GradTurbVar[iVar] -= Proj_Mean_GradTurbVar_Edge[iVar]*proj_vector_ij -
      (TurbVar_j[iVar]-TurbVar_i[iVar])*proj_vector_ij;
    }
  }

  FinishResidualCalc(config);

  AD::SetPreaccOut(Flux, nVar);
  AD::EndPreacc();

  return ResidualType<>(Flux, Jacobian_i, Jacobian_j);

}

CNEMOAvgGrad_TurbSA::CNEMOAvgGrad_TurbSA(unsigned short val_nDim, unsigned short val_nVar,
                                         unsigned short val_nPrimVar,
                                         unsigned short val_nPrimVarGrad,
                                         bool correct_gradient, const CConfig* config) :
                                         CNEMOAvgGrad_Scalar(val_nDim, val_nVar,val_nPrimVar,val_nPrimVarGrad,
                                                             correct_gradient, config) { }

void CNEMOAvgGrad_TurbSA::ExtraADPreaccIn() { }

void CNEMOAvgGrad_TurbSA::FinishResidualCalc(const CConfig* config) {

  /*--- Compute mean effective viscosity ---*/

  su2double nu_i = Laminar_Viscosity_i/Density_i;
  su2double nu_j = Laminar_Viscosity_j/Density_j;
  su2double nu_e = 0.5*(nu_i+nu_j+TurbVar_i[0]+TurbVar_j[0]);

  Flux[0] = nu_e*Proj_Mean_GradTurbVar[0]/sigma;

  /*--- For Jacobians -> Use of TSL approx. to compute derivatives of the gradients ---*/

  if (implicit) {
    Jacobian_i[0][0] = (0.5*Proj_Mean_GradTurbVar[0]-nu_e*proj_vector_ij)/sigma;
    Jacobian_j[0][0] = (0.5*Proj_Mean_GradTurbVar[0]+nu_e*proj_vector_ij)/sigma;
  }

}

CNEMOAvgGrad_TurbSA_Neg::CNEMOAvgGrad_TurbSA_Neg(unsigned short val_nDim, unsigned short val_nVar,
                                                 unsigned short val_nPrimVar,
                                                 unsigned short val_nPrimVarGrad,
                                                 bool correct_gradient, const CConfig* config) :
                                                 CNEMOAvgGrad_Scalar(val_nDim, val_nVar,val_nPrimVar,val_nPrimVarGrad,
                                                                     correct_gradient, config) { }

void CNEMOAvgGrad_TurbSA_Neg::ExtraADPreaccIn() { }

void CNEMOAvgGrad_TurbSA_Neg::FinishResidualCalc(const CConfig* config) {

  /*--- Compute mean effective viscosity ---*/

  su2double nu_i = Laminar_Viscosity_i/Density_i;
  su2double nu_j = Laminar_Viscosity_j/Density_j;

  su2double nu_ij = 0.5*(nu_i+nu_j);
  su2double nu_tilde_ij = 0.5*(TurbVar_i[0]+TurbVar_j[0]);

  su2double nu_e;

  if (nu_tilde_ij > 0.0) {
    nu_e = nu_ij + nu_tilde_ij;
  }
  else {
    su2double Xi = nu_tilde_ij/nu_ij;
    su2double fn = (cn1 + Xi*Xi*Xi)/(cn1 - Xi*Xi*Xi);
    nu_e = nu_ij + fn*nu_tilde_ij;
  }

  Flux[0] = nu_e*Proj_Mean_GradTurbVar_Normal[0]/sigma;

  /*--- For Jacobians -> Use of TSL approx. to compute derivatives of the gradients ---*/

  if (implicit) {
    Jacobian_i[0][0] = (0.5*Proj_Mean_GradTurbVar[0]-nu_e*proj_vector_ij)/sigma;
    Jacobian_j[0][0] = (0.5*Proj_Mean_GradTurbVar[0]+nu_e*proj_vector_ij)/sigma;
  }

}

CNEMOAvgGrad_TurbSST::CNEMOAvgGrad_TurbSST(unsigned short val_nDim, unsigned short val_nVar,
                                           unsigned short val_nPrimVar,
                                           unsigned short val_nPrimVarGrad,
                                           const su2double* constants, bool correct_grad,
                                           const CConfig* config) :
                                           CNEMOAvgGrad_Scalar(val_nDim, val_nVar,val_nPrimVar,val_nPrimVarGrad,
                                                               correct_grad, config),
  sigma_k1(constants[0]),
  sigma_k2(constants[1]),
  sigma_om1(constants[2]),
  sigma_om2(constants[3]) {

}

void CNEMOAvgGrad_TurbSST::ExtraADPreaccIn() {
  AD::SetPreaccIn(F1_i); AD::SetPreaccIn(F1_j);
}

void CNEMOAvgGrad_TurbSST::FinishResidualCalc(const CConfig* config) {

  su2double sigma_kine_i, sigma_kine_j, sigma_omega_i, sigma_omega_j;
  su2double diff_i_kine, diff_i_omega, diff_j_kine, diff_j_omega;

  /*--- Compute the blended constant for the viscous terms ---*/
  sigma_kine_i  = F1_i*sigma_k1  + (1.0 - F1_i)*sigma_k2;
  sigma_kine_j  = F1_j*sigma_k1  + (1.0 - F1_j)*sigma_k2;
  sigma_omega_i = F1_i*sigma_om1 + (1.0 - F1_i)*sigma_om2;
  sigma_omega_j = F1_j*sigma_om1 + (1.0 - F1_j)*sigma_om2;

  /*--- Compute mean effective viscosity ---*/
  diff_i_kine  = Laminar_Viscosity_i + sigma_kine_i*Eddy_Viscosity_i;
  diff_j_kine  = Laminar_Viscosity_j + sigma_kine_j*Eddy_Viscosity_j;
  diff_i_omega = Laminar_Viscosity_i + sigma_omega_i*Eddy_Viscosity_i;
  diff_j_omega = Laminar_Viscosity_j + sigma_omega_j*Eddy_Viscosity_j;

  su2double diff_kine  = 0.5*(diff_i_kine  + diff_j_kine);
  su2double diff_omega = 0.5*(diff_i_omega + diff_j_omega);

  Flux[0] = diff_kine*Proj_Mean_GradTurbVar[0];
  Flux[1] = diff_omega*Proj_Mean_GradTurbVar[1];

  /*--- For Jacobians -> Use of TSL approx. to compute derivatives of the gradients ---*/
  if (implicit) {
    su2double proj_on_rho = proj_vector_ij/Density_i;

    Jacobian_i[0][0] = -diff_kine*proj_on_rho;  Jacobian_i[0][1] = 0.0;
    Jacobian_i[1][0] = 0.0;                     Jacobian_i[1][1] = -diff_omega*proj_on_rho;

    proj_on_rho = proj_vector_ij/Density_j;

    Jacobian_j[0][0] = diff_kine*proj_on_rho;   Jacobian_j[0][1] = 0.0;
    Jacobian_j[1][0] = 0.0;                     Jacobian_j[1][1] = diff_omega*proj_on_rho;
  }

}