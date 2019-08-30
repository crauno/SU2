/*!
 * \file output_adj_heat.cpp
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

#include "../../include/output/CAdjHeatOutput.hpp"

#include "../../../Common/include/geometry_structure.hpp"
#include "../../include/solver_structure.hpp"

CAdjHeatOutput::CAdjHeatOutput(CConfig *config, unsigned short nDim) : COutput(config, nDim, false) {
  
  /*--- Set the default history fields if nothing is set in the config file ---*/
  
  if (nRequestedHistoryFields == 0){
    requestedHistoryFields.push_back("ITER");
    requestedHistoryFields.push_back("RMS_RES");
    requestedHistoryFields.push_back("SENSITIVITY");
    nRequestedHistoryFields = requestedHistoryFields.size();
  }
  
  if (nRequestedScreenFields == 0){
    if (multiZone) requestedScreenFields.push_back("OUTER_ITER");
    requestedScreenFields.push_back("INNER_ITER");    
    requestedScreenFields.push_back("RMS_ADJ_TEMPERATURE");
    requestedScreenFields.push_back("SENS_GEO");
    nRequestedScreenFields = requestedScreenFields.size();
  }
  
  if (nRequestedVolumeFields == 0){
    requestedVolumeFields.push_back("COORDINATES");
    requestedVolumeFields.push_back("SOLUTION");
    requestedVolumeFields.push_back("SENSITIVITY");
    nRequestedVolumeFields = requestedVolumeFields.size();
  }
  
  stringstream ss;
  ss << "Zone " << config->GetiZone() << " (Adj. Heat)";
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

  if (convField.size() == 0 ) convField = "RMS_ADJ_TEMPERATURE";

}

CAdjHeatOutput::~CAdjHeatOutput(void) {

  if (rank == MASTER_NODE){
    histFile.close();
  }

}

void CAdjHeatOutput::SetHistoryOutputFields(CConfig *config){
  
  /// BEGIN_GROUP: RMS_RES, DESCRIPTION: The root-mean-square residuals of the conservative variables. 
  /// DESCRIPTION: Root-mean square residual of the adjoint Pressure.
  AddHistoryOutput("RMS_ADJ_TEMPERATURE",    "rms[A_T]",  FORMAT_FIXED, "RMS_RES", "Root-mean square residual of the adjoint temperature.", TYPE_RESIDUAL); 
  /// END_GROUP
  
  /// BEGIN_GROUP: MAX_RES, DESCRIPTION: The maximum residuals of the conservative variables. 
  /// DESCRIPTION: Maximum residual of the adjoint Pressure.
  AddHistoryOutput("MAX_ADJ_TEMPERATURE",    "max[A_T]",  FORMAT_FIXED, "MAX_RES", "Maximum residual of the adjoint temperature.", TYPE_RESIDUAL);

  /// BEGIN_GROUP: MAX_RES, DESCRIPTION: The maximum residuals of the conservative variables. 
  /// DESCRIPTION: Maximum residual of the adjoint Pressure.
  AddHistoryOutput("MAX_ADJ_TEMPERATURE",    "max[A_T]",  FORMAT_FIXED, "BGS_RES", "BGS residual of the adjoint temperature.", TYPE_RESIDUAL);
  
  /// BEGIN_GROUP: SENSITIVITY, DESCRIPTION: Sensitivities of different geometrical or boundary values.   
  /// DESCRIPTION: Sum of the geometrical sensitivities on all markers set in MARKER_MONITORING.
  AddHistoryOutput("SENS_GEO",   "Sens_Geo",   FORMAT_SCIENTIFIC, "SENSITIVITY", "Sum of the geometrical sensitivities on all markers set in MARKER_MONITORING.", TYPE_COEFFICIENT); 
  /// END_GROUP
  
}

void CAdjHeatOutput::LoadHistoryData(CConfig *config, CGeometry *geometry, CSolver **solver) { 
  
  CSolver* adjheat_solver = solver[ADJHEAT_SOL];

  SetHistoryOutputValue("RMS_ADJ_TEMPERATURE", log10(adjheat_solver->GetRes_RMS(0)));
 
  SetHistoryOutputValue("MAX_ADJ_TEMPERATURE", log10(adjheat_solver->GetRes_Max(0)));
 
  if (multiZone)
    SetHistoryOutputValue("MAX_ADJ_TEMPERATURE", log10(adjheat_solver->GetRes_BGS(0)));
    
  SetHistoryOutputValue("SENS_GEO", adjheat_solver->GetTotal_Sens_Geo());
 

}

void CAdjHeatOutput::SetVolumeOutputFields(CConfig *config){
  
  // Grid coordinates
  AddVolumeOutput("COORD-X", "x", "COORDINATES", "x-component of the coordinate vector");
  AddVolumeOutput("COORD-Y", "y", "COORDINATES", "y-component of the coordinate vector");
  if (nDim == 3)
    AddVolumeOutput("COORD-Z", "z", "COORDINATES", "z-component of the coordinate vector");

  
  /// BEGIN_GROUP: CONSERVATIVE, DESCRIPTION: The conservative variables of the adjoint solver.
  /// DESCRIPTION: Adjoint Pressure.
  AddVolumeOutput("ADJ_TEMPERATURE",    "Adjoint_Temperature",    "SOLUTION" ,"Adjoint Temperature");
  /// END_GROUP
  
  
  /// BEGIN_GROUP: RESIDUAL, DESCRIPTION: Residuals of the conservative variables. 
  /// DESCRIPTION: Residual of the adjoint Pressure.
  AddVolumeOutput("RES_ADJ_TEMPERATURE",    "Residual_Adjoint_Temperature",    "RESIDUAL", "Residual of the Adjoint Temperature");
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

void CAdjHeatOutput::LoadVolumeData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint){
  
  CVariable* Node_AdjHeat = solver[ADJHEAT_SOL]->node[iPoint]; 
  CPoint*    Node_Geo     = geometry->node[iPoint];
  

  SetVolumeOutputValue("COORD-X", iPoint,  Node_Geo->GetCoord(0));  
  SetVolumeOutputValue("COORD-Y", iPoint,  Node_Geo->GetCoord(1));
  if (nDim == 3)
    SetVolumeOutputValue("COORD-Z", iPoint, Node_Geo->GetCoord(2));
  
  SetVolumeOutputValue("ADJ_TEMPERATURE",    iPoint, Node_AdjHeat->GetSolution(0));
  
  // Residuals
  SetVolumeOutputValue("RES_ADJ_TEMPERATURE", iPoint, Node_AdjHeat->GetSolution(0) - Node_AdjHeat->GetSolution_Old(0));
  
  SetVolumeOutputValue("SENSITIVITY-X", iPoint, Node_AdjHeat->GetSensitivity(0));
  SetVolumeOutputValue("SENSITIVITY-Y", iPoint, Node_AdjHeat->GetSensitivity(1));
  if (nDim == 3)
    SetVolumeOutputValue("SENSITIVITY-Z", iPoint, Node_AdjHeat->GetSensitivity(2));
  
}

void CAdjHeatOutput::LoadSurfaceData(CConfig *config, CGeometry *geometry, CSolver **solver, unsigned long iPoint, unsigned short iMarker, unsigned long iVertex){
  
  SetVolumeOutputValue("SENSITIVITY", iPoint, solver[ADJHEAT_SOL]->GetCSensitivity(iMarker, iVertex));
  
}


