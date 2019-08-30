/*!
 * \file output_adj_flow_inc.cpp
 * \brief Main subroutines for flow discrete adjoint output
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
 * Copyright 2012-2018, Francisco D. Palacios, Thomas D. Economon,
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

#include "../../include/output/CAdjFlowIncOutput.hpp"

#include "../../../Common/include/geometry_structure.hpp"
#include "../../include/solver_structure.hpp"

CAdjFlowIncOutput::CAdjFlowIncOutput(CConfig *config, unsigned short nDim) : COutput(config, nDim, false) {
  
  cont_adj = config->GetContinuous_Adjoint();
   
  turb_model = config->GetKind_Turb_Model();
  
  heat = config->GetEnergy_Equation();
  
  weakly_coupled_heat = config->GetWeakly_Coupled_Heat();
  
  /*--- Set the default history fields if nothing is set in the config file ---*/
  
  if (nRequestedHistoryFields == 0){
    requestedHistoryFields.push_back("ITER");
    requestedHistoryFields.push_back("RMS_RES");
    requestedHistoryFields.push_back("SENSITIVITY");
    nRequestedHistoryFields = requestedHistoryFields.size();
  }
  
  if (nRequestedScreenFields == 0){
    if (config->GetTime_Domain()) requestedScreenFields.push_back("TIME_ITER");    
    if (multiZone) requestedScreenFields.push_back("OUTER_ITER");
    requestedScreenFields.push_back("INNER_ITER");    
    requestedScreenFields.push_back("RMS_ADJ_PRESSURE");
    requestedScreenFields.push_back("RMS_ADJ_VELOCITY-X");
    requestedScreenFields.push_back("SENS_GEO");
    requestedScreenFields.push_back("SENS_AOA");
    nRequestedScreenFields = requestedScreenFields.size();
  }
  
  if (nRequestedVolumeFields == 0){
    requestedVolumeFields.push_back("COORDINATES");
    requestedVolumeFields.push_back("SOLUTION");    
    requestedVolumeFields.push_back("SENSITIVITY");
    nRequestedVolumeFields = requestedVolumeFields.size();
  }
  
  stringstream ss;
  ss << "Zone " << config->GetiZone() << " (Adj. Incomp. Fluid)";
  multiZoneHeaderString = ss.str();
  
  /*--- Set the volume filename --- */
  
  volumeFilename = config->GetAdj_FileName();
  
  /*--- Set the surface filename --- */
  
  surfaceFilename = config->GetSurfAdjCoeff_FileName();
  
  /*--- Set the restart filename --- */
  
  restartFilename = config->GetRestart_AdjFileName();
  
  /*--- Add the obj. function extension --- */
  
  restartFilename = config->GetObjFunc_Extension(restartFilename);

  /*--- Set the default convergence field --- */

  if (convField.size() == 0 ) convField = "RMS_ADJ_PRESSURE";

}

CAdjFlowIncOutput::~CAdjFlowIncOutput(void) {

  if (rank == MASTER_NODE){
    histFile.close();
  }

}

void CAdjFlowIncOutput::SetHistoryOutputFields(CConfig *config){
  
  /// BEGIN_GROUP: RMS_RES, DESCRIPTION: The root-mean-square residuals of the SOLUTION variables. 
  /// DESCRIPTION: Root-mean square residual of the adjoint Pressure.
  AddHistoryOutput("RMS_ADJ_PRESSURE",    "rms[A_P]",  FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint Pressure.", TYPE_RESIDUAL); 
  /// DESCRIPTION: Root-mean square residual of the adjoint Velocity x-component.
  AddHistoryOutput("RMS_ADJ_VELOCITY-X", "rms[A_U]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint Velocity x-component.", TYPE_RESIDUAL);
  /// DESCRIPTION: Root-mean square residual of the adjoint Velocity y-component.
  AddHistoryOutput("RMS_ADJ_VELOCITY-Y", "rms[A_V]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint Velocity y-component.", TYPE_RESIDUAL);
  /// DESCRIPTION: Root-mean square residual of the adjoint Velocity z-component.
  AddHistoryOutput("RMS_ADJ_VELOCITY-Z", "rms[A_W]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint Velocity z-component.", TYPE_RESIDUAL);
  /// DESCRIPTION: Maximum residual of the temperature.
  AddHistoryOutput("RMS_ADJ_HEAT", "rms[A_T]", FORMAT_FIXED, "RMS_RES", " Root-mean square residual of the adjoint temperature.", TYPE_RESIDUAL);
  if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      /// DESCRIPTION: Root-mean square residual of the adjoint nu tilde.
      AddHistoryOutput("RMS_ADJ_NU_TILDE", "rms[A_nu]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint nu tilde.", TYPE_RESIDUAL);      
      break;  
    case SST:
      /// DESCRIPTION: Root-mean square residual of the adjoint kinetic energy.
      AddHistoryOutput("RMS_ADJ_TKE", "rms[A_k]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint kinetic energy.", TYPE_RESIDUAL); 
      /// DESCRIPTION: Root-mean square residual of the adjoint dissipation.
      AddHistoryOutput("RMS_ADJ_DISSIPATION",    "rms[A_w]", FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint dissipation.", TYPE_RESIDUAL);   
      break;
    default: break;
    }
  }
  /// END_GROUP
  
  /// BEGIN_GROUP: MAX_RES, DESCRIPTION: The maximum residuals of the SOLUTION variables. 
  /// DESCRIPTION: Maximum residual of the adjoint Pressure.
  AddHistoryOutput("MAX_ADJ_PRESSURE",    "max[A_Rho]",  FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint Pressure.", TYPE_RESIDUAL);
  /// DESCRIPTION: Maximum residual of the adjoint Velocity x-component
  AddHistoryOutput("MAX_ADJ_VELOCITY-X", "max[A_RhoU]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint Velocity x-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: Maximum residual of the adjoint Velocity y-component
  AddHistoryOutput("MAX_ADJ_VELOCITY-Y", "max[A_RhoV]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint Velocity y-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: Maximum residual of the adjoint Velocity z-component
  AddHistoryOutput("MAX_ADJ_VELOCITY-Z", "max[A_RhoW]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint Velocity z-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: Maximum residual of the temperature.
  AddHistoryOutput("MAX_ADJ_HEAT", "max[A_T]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the temperature.", TYPE_RESIDUAL);
  if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      /// DESCRIPTION: Maximum residual of the adjoint nu tilde.
      AddHistoryOutput("MAX_ADJ_NU_TILDE", "max[A_nu]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint nu tilde.", TYPE_RESIDUAL);      
      break;  
    case SST:
      /// DESCRIPTION: Maximum residual of the adjoint kinetic energy.
      AddHistoryOutput("MAX_ADJ_TKE", "max[A_k]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint kinetic energy.", TYPE_RESIDUAL);   
      /// DESCRIPTION: Maximum residual of the adjoint dissipation.
      AddHistoryOutput("MAX_ADJ_DISSIPATION",    "max[A_w]", FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint dissipation.", TYPE_RESIDUAL); 
      break;
    default: break;
    }
  }
  /// END_GROUP
  
  /// BEGIN_GROUP: BGS_RES, DESCRIPTION: The block Gauss Seidel residuals of the SOLUTION variables. 
  /// DESCRIPTION: BGS residual of the adjoint Pressure.
  AddHistoryOutput("BGS_ADJ_PRESSURE",    "bgs[A_Rho]",  FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint Pressure.", TYPE_RESIDUAL);
  /// DESCRIPTION: BGS residual of the adjoint Velocity x-component
  AddHistoryOutput("BGS_ADJ_VELOCITY-X", "bsg[A_RhoU]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint Velocity x-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: BGS residual of the adjoint Velocity y-component
  AddHistoryOutput("BGS_ADJ_VELOCITY-Y", "bgs[A_RhoV]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint Velocity y-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: BGS residual of the adjoint Velocity z-component
  AddHistoryOutput("BGS_ADJ_VELOCITY-Z", "bgs[A_RhoW]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint Velocity z-component", TYPE_RESIDUAL); 
  /// DESCRIPTION: BGS residual of the temperature.
  AddHistoryOutput("BGS_ADJ_HEAT", "bgs[A_T]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint temperature.", TYPE_RESIDUAL);
  if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      /// DESCRIPTION: BGS residual of the adjoint nu tilde.
      AddHistoryOutput("BGS_ADJ_NU_TILDE", "bsg[A_nu]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint nu tilde.", TYPE_RESIDUAL);      
      break;  
    case SST:
      /// DESCRIPTION: BGS residual of the adjoint kinetic energy.
      AddHistoryOutput("BGS_ADJ_TKE", "bgs[A_k]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint kinetic energy.", TYPE_RESIDUAL);   
      /// DESCRIPTION: BGS residual of the adjoint dissipation.
      AddHistoryOutput("BGS_ADJ_DISSIPATION",    "bgs[A_w]", FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint dissipation.", TYPE_RESIDUAL); 
      break;
    default: break;
    }
  }
  /// END_GROUP
  
  
  /// BEGIN_GROUP: SENSITIVITY, DESCRIPTION: Sensitivities of different geometrical or boundary values.   
  /// DESCRIPTION: Sum of the geometrical sensitivities on all markers set in MARKER_MONITORING.
  AddHistoryOutput("SENS_GEO",   "Sens_Geo",   FORMAT_SCIENTIFIC, "SENSITIVITY", "Sum of the geometrical sensitivities on all markers set in MARKER_MONITORING.", TYPE_COEFFICIENT); 
  /// DESCRIPTION: Sensitivity of the objective function with respect to the far-field pressure.
  AddHistoryOutput("SENS_PRESS", "Sens_Press", FORMAT_SCIENTIFIC, "SENSITIVITY", "Sensitivity of the objective function with respect to the far-field pressure.", TYPE_COEFFICIENT); 
  /// DESCRIPTION: Sensitivity of the objective function with respect to the far-field temperature.
  AddHistoryOutput("SENS_TEMP",  "Sens_Temp",  FORMAT_SCIENTIFIC, "SENSITIVITY", " Sensitivity of the objective function with respect to the far-field temperature.", TYPE_COEFFICIENT); 
  /// DESCRIPTION: Sensitivity of the objective function with respect to the inlet velocity.
  AddHistoryOutput("SENS_VEL_IN", "Sens_Vin", FORMAT_SCIENTIFIC, "SENSITIVITY", " Sensitivity of the objective function with respect to the inlet velocity.", TYPE_COEFFICIENT); 
  /// DESCRIPTION: Sensitivity of the objective function with respect to the outlet pressure.
  AddHistoryOutput("SENS_PRESS_OUT",  "Sens_Pout",  FORMAT_SCIENTIFIC, "SENSITIVITY", "Sensitivity of the objective function with respect to the outlet pressure.", TYPE_COEFFICIENT); 
  /// END_GROUP
  
}

void CAdjFlowIncOutput::LoadHistoryData(CConfig *config, CGeometry *geometry, CSolver **solver) { 
  
  CSolver* adjflow_solver = solver[ADJFLOW_SOL];
  CSolver* adjturb_solver = solver[ADJTURB_SOL];  
  CSolver* adjheat_solver = solver[ADJHEAT_SOL];

  SetHistoryOutputValue("RMS_ADJ_PRESSURE", log10(adjflow_solver->GetRes_RMS(0)));
  SetHistoryOutputValue("RMS_ADJ_VELOCITY-X", log10(adjflow_solver->GetRes_RMS(1)));
  SetHistoryOutputValue("RMS_ADJ_VELOCITY-Y", log10(adjflow_solver->GetRes_RMS(2)));
  if (nDim == 3) {
    SetHistoryOutputValue("RMS_ADJ_VELOCITY-Z", log10(adjflow_solver->GetRes_RMS(3)));
  }
  if (weakly_coupled_heat){
    SetHistoryOutputValue("RMS_ADJ_HEAT",         log10(adjheat_solver->GetRes_RMS(0)));
  }
  if (heat){
    if (nDim == 3) SetHistoryOutputValue("RMS_ADJ_HEAT",         log10(adjflow_solver->GetRes_RMS(4)));
    else           SetHistoryOutputValue("RMS_ADJ_HEAT",         log10(adjflow_solver->GetRes_RMS(3)));
  }
  if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      SetHistoryOutputValue("RMS_ADJ_NU_TILDE", log10(adjturb_solver->GetRes_RMS(0)));
      break;  
    case SST:
      SetHistoryOutputValue("RMS_ADJ_TKE", log10(adjturb_solver->GetRes_RMS(0)));
      SetHistoryOutputValue("RMS_ADJ_DISSIPATION",    log10(adjturb_solver->GetRes_RMS(1)));
      break;
    default: break;
    }
  }
  SetHistoryOutputValue("MAX_ADJ_PRESSURE", log10(adjflow_solver->GetRes_Max(0)));
  SetHistoryOutputValue("MAX_ADJ_VELOCITY-X", log10(adjflow_solver->GetRes_Max(1)));
  SetHistoryOutputValue("MAX_ADJ_VELOCITY-Y", log10(adjflow_solver->GetRes_Max(2)));
  if (nDim == 3) {
    SetHistoryOutputValue("MAX_ADJ_VELOCITY-Z", log10(adjflow_solver->GetRes_Max(3)));
  }
  if (weakly_coupled_heat){
    SetHistoryOutputValue("MAX_ADJ_HEAT",         log10(adjheat_solver->GetRes_Max(0)));
  }
  if (heat){
    if (nDim == 3) SetHistoryOutputValue("MAX_ADJ_HEAT",         log10(adjflow_solver->GetRes_Max(4)));
    else           SetHistoryOutputValue("MAX_ADJ_HEAT",         log10(adjflow_solver->GetRes_Max(3)));
  }
  if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      SetHistoryOutputValue("MAX_ADJ_NU_TILDE", log10(adjturb_solver->GetRes_Max(0)));
      break;  
    case SST:
      SetHistoryOutputValue("MAX_ADJ_TKE", log10(adjturb_solver->GetRes_Max(0)));
      SetHistoryOutputValue("MAX_ADJ_DISSIPATION",    log10(adjturb_solver->GetRes_Max(1)));
      break;
    default: break;
    }
  }
  
  if (multiZone){
    SetHistoryOutputValue("BGS_ADJ_PRESSURE", log10(adjflow_solver->GetRes_BGS(0)));
    SetHistoryOutputValue("BGS_ADJ_VELOCITY-X", log10(adjflow_solver->GetRes_BGS(1)));
    SetHistoryOutputValue("BGS_ADJ_VELOCITY-Y", log10(adjflow_solver->GetRes_BGS(2)));
    if (nDim == 3) {
      SetHistoryOutputValue("BGS_ADJ_VELOCITY-Z", log10(adjflow_solver->GetRes_BGS(3)));
    }
    if (weakly_coupled_heat){
      SetHistoryOutputValue("BGS_ADJ_HEAT",         log10(adjheat_solver->GetRes_BGS(0)));
    }
    if (heat){
      if (nDim == 3) SetHistoryOutputValue("BGS_ADJ_HEAT",         log10(adjflow_solver->GetRes_BGS(4)));
      else           SetHistoryOutputValue("BGS_ADJ_HEAT",         log10(adjflow_solver->GetRes_BGS(3)));
    }
    if ((!config->GetFrozen_Visc_Disc() && !cont_adj) || (!config->GetFrozen_Visc_Cont() && cont_adj)){
      switch(turb_model){
      case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
        SetHistoryOutputValue("BGS_ADJ_NU_TILDE", log10(adjturb_solver->GetRes_BGS(0)));
        break;  
      case SST:
        SetHistoryOutputValue("BGS_ADJ_TKE", log10(adjturb_solver->GetRes_BGS(0)));
        SetHistoryOutputValue("BGS_ADJOINT_DISSIPATION",    log10(adjturb_solver->GetRes_BGS(1)));
        break;
      default: break;
      }
    }
  }
  
  SetHistoryOutputValue("SENS_GEO", adjflow_solver->GetTotal_Sens_Geo());
  SetHistoryOutputValue("SENS_PRESS", adjflow_solver->GetTotal_Sens_Press());
  SetHistoryOutputValue("SENS_TEMP", adjflow_solver->GetTotal_Sens_Temp());
  SetHistoryOutputValue("SENS_VEL_IN", adjflow_solver->GetTotal_Sens_ModVel());
  SetHistoryOutputValue("SENS_PRESS_OUT", adjflow_solver->GetTotal_Sens_BPress());

}

void CAdjFlowIncOutput::SetVolumeOutputFields(CConfig *config){
  
  
  // Grid coordinates
  AddVolumeOutput("COORD-X", "x", "COORDINATES", "x-component of the coordinate vector");
  AddVolumeOutput("COORD-Y", "y", "COORDINATES", "y-component of the coordinate vector");
  if (nDim == 3)
    AddVolumeOutput("COORD-Z", "z", "COORDINATES", "z-component of the coordinate vector");
  
  /// BEGIN_GROUP: SOLUTION, DESCRIPTION: The SOLUTION variables of the adjoint solver.
  /// DESCRIPTION: Adjoint Pressure.
  AddVolumeOutput("ADJ_PRESSURE",    "Adjoint_Pressure",    "SOLUTION", "Adjoint pressure"); 
  /// DESCRIPTION: Adjoint Velocity x-component.
  AddVolumeOutput("ADJ_VELOCITY-X", "Adjoint_Velocity_x", "SOLUTION", "x-component of the adjoint velocity vector"); 
  /// DESCRIPTION: Adjoint Velocity y-component.
  AddVolumeOutput("ADJ_VELOCITY-Y", "Adjoint_Velocity_y", "SOLUTION", "y-component of the adjoint velocity vector"); 
  if (nDim == 3)
    /// DESCRIPTION: Adjoint Velocity z-component.
    AddVolumeOutput("ADJ_VELOCITY-Z", "Adjoint_Velocity_z", "SOLUTION", "z-component of the adjoint velocity vector"); 
 
  if (weakly_coupled_heat || heat){
    AddVolumeOutput("ADJ_HEAT", "Adjoint_Heat", "SOLUTION",  "Adjoint heat");
  }

  if (!config->GetFrozen_Visc_Disc()){    
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      /// DESCRIPTION: Adjoint nu tilde.
      AddVolumeOutput("ADJ_NU_TILDE", "Adjoint_Nu_Tilde", "SOLUTION", "Adjoint Spalart-Allmaras variable"); 
      break;  
    case SST:
      /// DESCRIPTION: Adjoint kinetic energy.
      AddVolumeOutput("ADJ_TKE", "Adjoint_TKE", "SOLUTION", "Adjoint turbulent kinetic energy"); 
      /// DESCRIPTION: Adjoint dissipation.
      AddVolumeOutput("ADJ_DISSIPATION", "Adjoint_Omega", "SOLUTION", "Adjoint rate of dissipation");  
      break;
    default: break;
    }
  }
  /// END_GROUP
  
  // Grid velocity
  if (config->GetGrid_Movement()){
    AddVolumeOutput("GRID_VELOCITY-X", "Grid_Velocity_x", "GRID_VELOCITY", "x-component of the grid velocity vector");
    AddVolumeOutput("GRID_VELOCITY-Y", "Grid_Velocity_y", "GRID_VELOCITY", "y-component of the grid velocity vector");
    if (nDim == 3 ) 
      AddVolumeOutput("GRID_VELOCITY-Z", "Grid_Velocity_z", "GRID_VELOCITY", "z-component of the grid velocity vector");
  }
  
  /// BEGIN_GROUP: RESIDUAL, DESCRIPTION: Residuals of the SOLUTION variables. 
  /// DESCRIPTION: Residual of the adjoint Pressure.
  AddVolumeOutput("RES_ADJ_PRESSURE",    "Residual_Adjoint_Pressure",    "RESIDUAL", "Residual of the adjoint pressure");  
  /// DESCRIPTION: Residual of the adjoint Velocity x-component.
  AddVolumeOutput("RES_ADJ_VELOCITY-X", "Residual_Adjoint_Velocity_x", "RESIDUAL", "Residual of the adjoint x-velocity");  
  /// DESCRIPTION: Residual of the adjoint Velocity y-component.
  AddVolumeOutput("RES_ADJ_VELOCITY-Y", "Residual_Adjoint_Velocity_y", "RESIDUAL", "Residual of the adjoint y-velocity");  
  if (nDim == 3)
    /// DESCRIPTION: Residual of the adjoint Velocity z-component.
    AddVolumeOutput("RES_ADJ_Velocity-Z", "Residual_Adjoint_Velocity_z", "RESIDUAL", "Residual of the adjoint z-velocity"); 
  /// DESCRIPTION: Residual of the adjoint energy. 
  AddVolumeOutput("RES_ADJ_HEAT", "Residual_Adjoint_Heat", "RESIDUAL", "Residual of the adjoint heat");            
  if (!config->GetFrozen_Visc_Disc()){      
    switch(turb_model){
    case SA: case SA_NEG: case SA_E: case SA_COMP: case SA_E_COMP:
      /// DESCRIPTION: Residual of the nu tilde. 
      AddVolumeOutput("RES_ADJ_NU_TILDE", "Residual_Adjoint_Nu_Tilde", "RESIDUAL", "Residual of the adjoint Spalart-Allmaras variable"); 
      break;  
    case SST:
      /// DESCRIPTION: Residual of the adjoint kinetic energy. 
      AddVolumeOutput("RES_ADJ_TKE", "Residual_Adjoint_TKE", "RESIDUAL", "Residual of the adjoint turb. kinetic energy");    
      /// DESCRIPTION: Residual of the adjoint dissipation.
      AddVolumeOutput("RES_ADJ_DISSIPATION", "Residual_Adjoint_Omega", "RESIDUAL", "Residual of adjoint rate of dissipation");
      break;
    default: break;
    }
  }
  /// END_GROUP
  
  /// BEGIN_GROUP: SENSITIVITY, DESCRIPTION: Geometrical sensitivities of the current objective function.
  /// DESCRIPTION: Sensitivity x-component.
  AddVolumeOutput("SENSITIVITY-X", "Sensitivity_x", "SENSITIVITY", "x-component of the sensitivity vector"); 
  /// DESCRIPTION: Sensitivity y-component.
  AddVolumeOutput("SENSITIVITY-Y", "Sensitivity_y", "SENSITIVITY", "y-component of the sensitivity vector");
  if (nDim == 3)
    /// DESCRIPTION: Sensitivity z-component.
    AddVolumeOutput("SENSITIVITY-Z", "Sensitivity_z", "SENSITIVITY", "z-component of the sensitivity vector");   
  /// DESCRIPTION: Sensitivity in normal direction.
  AddVolumeOutput("SENSITIVITY", "Surface_Sensitivity", "SENSITIVITY", "sensitivity in normal direction"); 
  /// END_GROUP
 
}

void CAdjFlowIncOutput::LoadVolumeData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint){
  
  CVariable* Node_AdjFlow = solver[ADJFLOW_SOL]->node[iPoint]; 
  CVariable* Node_AdjHeat = NULL;
  CVariable* Node_AdjTurb = NULL;
  CPoint*    Node_Geo     = geometry->node[iPoint];
  
  if (config->GetKind_Turb_Model() != NONE && !config->GetFrozen_Visc_Disc()){
    Node_AdjTurb = solver[ADJTURB_SOL]->node[iPoint]; 
  }
  if (weakly_coupled_heat){
    Node_AdjHeat = solver[ADJHEAT_SOL]->node[iPoint];
  }
  
  SetVolumeOutputValue("COORD-X", iPoint,  Node_Geo->GetCoord(0));  
  SetVolumeOutputValue("COORD-Y", iPoint,  Node_Geo->GetCoord(1));
  if (nDim == 3)
    SetVolumeOutputValue("COORD-Z", iPoint, Node_Geo->GetCoord(2));
  
  SetVolumeOutputValue("ADJ_PRESSURE",    iPoint, Node_AdjFlow->GetSolution(0));
  SetVolumeOutputValue("ADJ_VELOCITY-X", iPoint, Node_AdjFlow->GetSolution(1));
  SetVolumeOutputValue("ADJ_VELOCITY-Y", iPoint, Node_AdjFlow->GetSolution(2));
  if (nDim == 3){
    SetVolumeOutputValue("ADJ_VELOCITY-Z", iPoint, Node_AdjFlow->GetSolution(3));
  }

  if (weakly_coupled_heat){
    SetVolumeOutputValue("ADJ_HEAT", iPoint, Node_AdjHeat->GetSolution(0));
  }
  if (heat){
    if (nDim == 3) SetVolumeOutputValue("ADJ_HEAT", iPoint, Node_AdjFlow->GetSolution(4));
    else           SetVolumeOutputValue("ADJ_HEAT", iPoint, Node_AdjFlow->GetSolution(3));
  }
  // Turbulent 
  if (!config->GetFrozen_Visc_Disc()){         
    switch(turb_model){
    case SST:
      SetVolumeOutputValue("ADJ_TKE", iPoint, Node_AdjTurb->GetSolution(0));
      SetVolumeOutputValue("ADJ_DISSIPATION", iPoint, Node_AdjTurb->GetSolution(1));
      break;
    case SA: case SA_COMP: case SA_E: 
    case SA_E_COMP: case SA_NEG: 
      SetVolumeOutputValue("ADJ_NU_TILDE", iPoint, Node_AdjTurb->GetSolution(0));
      break;
    case NONE:
      break;
    }
  }
  
  // Residuals
  SetVolumeOutputValue("RES_ADJ_PRESSURE", iPoint, Node_AdjFlow->GetSolution(0) - Node_AdjFlow->GetSolution_Old(0));
  SetVolumeOutputValue("RES_ADJ_VELOCITY-X", iPoint, Node_AdjFlow->GetSolution(1) - Node_AdjFlow->GetSolution_Old(1));
  SetVolumeOutputValue("RES_ADJ_VELOCITY-Y", iPoint, Node_AdjFlow->GetSolution(2) - Node_AdjFlow->GetSolution_Old(2));
  if (nDim == 3){
    SetVolumeOutputValue("RES_ADJ_VELOCITY-Z", iPoint, Node_AdjFlow->GetSolution(3) - Node_AdjFlow->GetSolution_Old(3));
  } 
  if (!config->GetFrozen_Visc_Disc()){            
    switch(config->GetKind_Turb_Model()){
    case SST:
      SetVolumeOutputValue("RES_ADJ_TKE", iPoint, Node_AdjTurb->GetSolution(0) - Node_AdjTurb->GetSolution_Old(0));
      SetVolumeOutputValue("RES_ADJ_DISSIPATION", iPoint, Node_AdjTurb->GetSolution(1) - Node_AdjTurb->GetSolution_Old(1));
      break;
    case SA: case SA_COMP: case SA_E: 
    case SA_E_COMP: case SA_NEG: 
      SetVolumeOutputValue("RES_ADJ_NU_TILDE", iPoint, Node_AdjTurb->GetSolution(0) - Node_AdjTurb->GetSolution_Old(0));
      break;
    case NONE:
      break;
    }
  }
  
  SetVolumeOutputValue("SENSITIVITY-X", iPoint, Node_AdjFlow->GetSensitivity(0));
  SetVolumeOutputValue("SENSITIVITY-Y", iPoint, Node_AdjFlow->GetSensitivity(1));
  if (nDim == 3)
    SetVolumeOutputValue("SENSITIVITY-Z", iPoint, Node_AdjFlow->GetSensitivity(2));
  
}

void CAdjFlowIncOutput::LoadSurfaceData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint, unsigned short iMarker, unsigned long iVertex){
  
  SetVolumeOutputValue("SENSITIVITY", iPoint, solver[ADJFLOW_SOL]->GetCSensitivity(iMarker, iVertex));
  
}


bool CAdjFlowIncOutput::SetInit_Residuals(CConfig *config){
  
  return (config->GetTime_Marching() != STEADY && (curInnerIter == 0))|| 
        (config->GetTime_Marching() == STEADY && (curTimeIter < 2)); 
  
}

bool CAdjFlowIncOutput::SetUpdate_Averages(CConfig *config){
  return false;
  
//  return (config->GetUnsteady_Simulation() != STEADY && !dualtime);
      
}

