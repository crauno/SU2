#!/usr/bin/env python

## \file NITRO_Tester.py
#  \brief NITRO Tester solver (for the NITRO approach involving forced moving boundary condition) used for testing the Py wrapper for external FSI coupling.
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

# Imports from pyBeam (need to be reviewed)
from pyBeamIO import pyBeamConfig as pyConfig
from pyBeamIO import pyBeamInput as pyInput
import pyBeam

# ----------------------------------------------------------------------
#  Beam object
# ----------------------------------------------------------------------


class pyBeamSolver:
  """Description"""

  def __init__(self, config_fileName):
    """ Description. """

    self.Config_file = config_fileName
    self.Config = {}

    print("\n------------------------------ Configuring the structural tester solver for FSI simulation: pyBeam ------------------------------")
    # Parsing config file
    self.Config = pyConfig.pyBeamConfig(config_fileName)  # Beam configuration file

    self.Mesh_file = self.Config['B_MESH']
    self.Property = self.Config['B_PROPERTY']

    # Parsing mesh file
    self.nDim= pyInput.readDimension(self.Config['B_MESH'])
    self.node_py, self.nPoint = pyInput.readMesh(self.Config['B_MESH'], self.nDim)
    self.elem_py, self.nElem = pyInput.readConnectivity(self.Config['B_MESH'])
    self.Constr, self.nConstr = pyInput.readConstr(self.Config['B_MESH'])
    self.RBE2_py, self.nRBE2 = pyInput.readRBE2(self.Config['B_MESH'])
    # Parsing Property file
    self.Prop, self.nProp = pyInput.readProp(self.Config['B_PROPERTY'])

    # Initializing objects
    self.beam = pyBeam.CBeamSolver()
    self.inputs = pyBeam.CInput(self.nPoint, self.nElem, self.nRBE2)

    # Sending to CInput object
    pyConfig.parseInput(self.Config, self.inputs, self.Constr, self.nConstr)
    # Assigning input values to the input object in C++
    self.inputs.SetParameters()
    # Initialize the input in the beam solver
    self.beam.InitializeInput(self.inputs)

    # Assigning values to the CNode objects in C++
    self.node = []
    for i in range(self.nPoint):
        self.node.append(pyBeam.CNode(self.node_py[i].GetID()))
        for j in range(self.nDim):
            self.node[i].SetCoordinate(j, float(self.node_py[i].GetCoord()[j][0]))
            self.node[i].SetCoordinate0(j, float(self.node_py[i].GetCoord0()[j][0]))
        self.beam.InitializeNode(self.node[i], i)

    # Assigning property values to the property objects in C++
    self.beam_prop = []
    for i in range(self.nProp):
        self.beam_prop.append(pyBeam.CProperty(i))
        self.beam_prop[i].SetSectionProperties(self.Prop[i].GetA(), self.Prop[i].GetIyy(), self.Prop[i].GetIzz(), self.Prop[i].GetJt())

    # Assigning element values to the element objects in C++
    self.element = []
    for i in range(self.nElem):
        self.element.append(pyBeam.CElement(i))
        self.element[i].Initializer(self.node[self.elem_py[i].GetNodes()[0, 0] - 1], self.node[self.elem_py[i].GetNodes()[1, 0] - 1],
                               self.beam_prop[self.elem_py[i].GetProperty() - 1], self.inputs, self.elem_py[i].GetAuxVector()[0, 0],
                               self.elem_py[i].GetAuxVector()[1, 0], self.elem_py[i].GetAuxVector()[2, 0])
        self.beam.InitializeElement(self.element[i], i)

    print("\n------------------------------ Reading the SU2 mesh (is it needed though?)------------------------------")

    # Here we need to pass the AeroPoint matrix of the wing grid
    # IF ANY, assigning RBE2_element values to the RBE2 objects in C++
    if self.nRBE2 != 0:
        self.RBE2 = []
        for i in range(self.nRBE2):
            self.RBE2.append(pyBeam.CRBE2(i))
            self.RBE2[i].Initializer(self.node[self.RBE2_py[i].GetNodes()[0, 0] - 1], self.node[self.RBE2_py[i].GetNodes()[1, 0] - 1])
            self.beam.InitializeRBE2(self.RBE2[i], i)

    # finally intializing the structure for the solver
    self.beam.InitializeStructure()

    print("\n------------------------------ pyBeam initialization Done ----------------------------------------------")


    # What do we need yet?
    # run
    #  getDsiplacement
    #  Update Coordinates
    #  SetForces
    #  writeOutput (paraview tecplot etc)
    #

  '''
  What I need: 	          
  SolidSolver.staticComputation()
                else:
                    SolidSolver.run(0.0, 0.05)
  	      SolidSolver.writeSolution(0.0, self.FSIIter, Iter, NbIter)		
  	                    SolidSolver.updateSolution()
  	                    setLoad
  	                    OutputDisplacements
  '''

  def SetLoads(self, iVertex, loadX, loadY, loadZ):

    """ This function sets the load  """
    self.beam.SetLoads(iVertex, 0, loadX)
    self.beam.SetLoads(iVertex, 1, loadY)
    self.beam.SetLoads(iVertex, 2, loadZ)

  def getInitialCoordinates(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    coordX = self.beam.ExtractCoordinate0(iVertex, 0)
    coordY = self.beam.ExtractCoordinate0(iVertex, 1)
    coordZ = self.beam.ExtractCoordinate0(iVertex, 2)

    return coordX, coordY, coordZ

  def ExtractDisplacements(self,iVertex):

    """ This function returns the initial coordinates of the structural beam model  """
    dispX = self.beam.ExtractDisplacements(iVertex, 0)
    dispY = self.beam.ExtractDisplacements(iVertex, 1)
    dispZ = self.beam.ExtractDisplacements(iVertex, 2)

    return dispX, dispY, dispZ


  def run(self):
      "This function runs the solver. Needs to be run after __SetLoads"
      self.beam.Solve(0)


  def OutputDisplacements(self):
      "This function gives back the displacements on the nodes"










