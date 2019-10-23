#!/usr/bin/env python

## \file NITRO_Tester.py
#  \brief NITRO Tester solver (for the NITRO approach involving forced moving boundary condition).
#  \author Rocco Bombardieri, Ruben Sanchez
#  \version 5.0.0 "Raven"
#
# SU2 Original Developers: Dr. Francisco D. Palacios.
#                          Dr. Thomas D. Economon.
#
# SU2 Developers: Prof. Juan J. Alonso's group at Stanford University.
#                 Prof. Piero Colonna's group at Delft University of Technology.
#                 Prof. Nicolas R. Gauger's group at Kaiserslautern University of Technology.
#                 Prof. Alberto Guardone's group at Polytechnic University of Milan.
#                 Prof. Rafael Palacios' group at Imperial College London.
#                 Prof. Edwin van der Weide's group at the University of Twente.
#                 Prof. Vincent Terrapon's group at the University of Liege.
#
# Copyright (C) 2012-2017 SU2, the open-source CFD code.
#
# SU2 is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# SU2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SU2. If not, see <http://www.gnu.org/licenses/>.

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import sys, os

# ----------------------------------------------------------------------
#  Beam object
# ----------------------------------------------------------------------


class NITRO:
  """Description"""

  def __init__(self, config_fileName):
    """ Description. """

    self.Config_file = config_fileName
    self.Config = {}

    print("\n------------------------------ Configuring the structural tester solver for FSI simulation: pyBeam ------------------------------")

    # Load testcase directory directory
    file_dir = os.getcwd()

    self.beam = pyBeamSolver_pyBeam(file_dir, self.Config_file)

    # Some intermediate variables
    self.nPoint = self.beam.nPoint


  def SetLoads(self, iVertex, loadX, loadY, loadZ):

    """ This function sets the load  """
    self.beam.SetLoads(iVertex, loadX, loadY, loadZ)


  def GetInitialCoordinates(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    coordX, coordY, coordZ = self.beam.GetInitialCoordinates(iVertex)

    return coordX, coordY, coordZ

  def ExtractDisplacements(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    dispX, dispY, dispZ = self.beam.ExtractDisplacements(iVertex)

    return dispX, dispY, dispZ


  def run(self):
      "This function runs the solver. Needs to be run after __SetLoads"

      if self.beam.Config['RESTART'] == 1:
        self.beam.ReadRestart()
        self.beam.Restart()
      else:
        self.beam.Run()


  def OutputDisplacements(self):
      "This function gives back the displacements on the nodes"



class pyBeamADSolver:
  """Description"""

  def __init__(self, config_fileName):
    """ Description. """

    self.Config_file = config_fileName
    self.Config = {}

    print("\n----------- Configuring the AD solver in pyBeam --------------------")

    # Load testcase directory directory
    file_dir = os.getcwd()

    self.beam = pyBeamSolverAD_pyBeam(file_dir, self.Config_file)

    # Some intermediate variables
    self.nPoint = self.beam.nPoint

    self.sens_file = open("sensitivity_E.dat", "w")
    self.sens_file.write("Sensitivity E\n")
    self.sens_file.close()


  def SetLoads(self, iVertex, loadX, loadY, loadZ):

    """ This function sets the load  """
    self.beam.SetLoads(iVertex, loadX, loadY, loadZ)

  def SetDisplacementAdjoint(self, iVertex, adjX, adjY, adjZ):

    """ This function sets the displacement adjoint  """
    self.beam.SetDisplacementAdjoint(iVertex, adjX, adjY, adjZ)

  def GetLoadAdjoint(self, iVertex):

      """ This function gets the displacement adjoint  """
      sensX, sensY, sensZ = self.beam.GetLoadSensitivity(iVertex)
      return sensX, sensY, sensZ

  def RecordSolver(self):
      "This function completes the pyBeam recording. Needs to be run after __SetLoads"

      self.beam.ReadRestart()
      self.beam.StartRecording()
      self.beam.SetDependencies()
      self.beam.Restart()
      self.beam.StopRecording()

  def GetInitialCoordinates(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    coordX, coordY, coordZ = self.beam.GetInitialCoordinates(iVertex)

    return coordX, coordY, coordZ

  def ExtractDisplacements(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    dispX, dispY, dispZ = self.beam.ExtractDisplacements(iVertex)

    return dispX, dispY, dispZ


  def RunAdjoint(self):
      "This function runs the solver. Needs to be run after __SetLoads"

      self.beam.ComputeAdjoint()
      sens_E = self.beam.PrintSensitivityE()
      self.sens_file = open("sensitivity_E.dat", "a")
      self.sens_file.write(str(sens_E) + "\n")
      self.sens_file.close()

  def OutputDisplacements(self):
      "This function gives back the displacements on the nodes"








