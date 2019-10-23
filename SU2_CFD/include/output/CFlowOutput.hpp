/*!
 * \file CFlowOutput.hpp
 * \brief  Headers of the flow output.
 * \author F. Palacios, T. Economon, M. Colonno
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

#pragma once

#include "COutput.hpp"
#include "../variables/CVariable.hpp"

class CFlowOutput : public COutput{
public:
  /*!
   * \brief Constructor of the class
   * \param[in] config - Definition of the particular problem.
   */
  CFlowOutput(CConfig *config, unsigned short nDim, bool femOutput);

  /*!
   * \brief Destructor of the class.
   */
  ~CFlowOutput(void) override;

protected:
  /*!
   * \brief Add flow surface output fields
   * \param[in] config - Definition of the particular problem.
   */
  void AddAnalyzeSurfaceOutput(CConfig *config);
  
  /*!
   * \brief Set flow surface output field values
   * \param[in] solver - The container holding all solution data.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] output - Boolean indicating whether information should be written to screen
   */
  void SetAnalyzeSurface(CSolver *solver, CGeometry *geometry, CConfig *config, bool output);
  
  /*!
   * \brief Add aerodynamic coefficients as output fields
   * \param[in] config - Definition of the particular problem.
   */
  void AddAerodynamicCoefficients(CConfig *config);
  
  /*!
   * \brief  Set the value of the aerodynamic coefficients 
   * \param[in] config - Definition of the particular problem.
   * \param[in] flow_solver - The container holding all solution data.
   */
  void SetAerodynamicCoefficients(CConfig *config, CSolver *flow_solver);
  
  /*!
   * \brief Add CP inverse design output as history fields
   * \param[in] config - Definition of the particular problem.
   */
  void Add_CpInverseDesignOutput(CConfig *config);
  
  /*!
   * \brief Set CP inverse design output field values
   * \param[in] solver - The container holding all solution data.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void Set_CpInverseDesign(CSolver *solver, CGeometry *geometry, CConfig *config);
  
  /*!
   * \brief Compute value of the Q criteration for vortex idenfitication
   * \param[in] VelocityGradient - Velocity gradients
   * \return Value of the Q criteration at the node
   */
  su2double GetQ_Criterion(su2double** VelocityGradient) const;
  
  /*!
   * \brief Write information to meta data file
   * \param[in] output - Container holding the output instances per zone.   
   * \param[in] config - Definition of the particular problem per zone.
   */
  void WriteMetaData(CConfig *config, CGeometry *geometry);
  
  /*!
   * \brief Write any additional files defined for the current solver.
   * \param[in] config - Definition of the particular problem per zone.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - The container holding all solution data.
   */
  void WriteAdditionalFiles(CConfig *config, CGeometry *geometry, CSolver **solver_container) override;
  
  /*!
   * \brief Determines if the the volume output should be written.
   * \param[in] config - Definition of the particular problem.
   * \param[in] Iter - Current iteration index.
   */
  bool WriteVolume_Output(CConfig *config, unsigned long Iter) override;
  
  /*!
   * \brief Write the forces breakdown file
   * \param[in] config - Definition of the particular problem per zone.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver_container - The container holding all solution data.
   */
  void WriteForcesBreakdown(CConfig *config, CGeometry *geometry, CSolver **solver_container);
  
  /*!
   * \brief Set the time averaged output fields.
   */
  void SetTimeAveragedFields();
  
  /*!
   * \brief Load the time averaged output fields.
   * \param iPoint
   * \param node_flow
   */
  void LoadTimeAveragedData(unsigned long iPoint, CVariable *node_flow);
};
