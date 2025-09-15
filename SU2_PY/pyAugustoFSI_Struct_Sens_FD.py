#!/usr/bin/env python3

## \file fsi_computation.py
#  \brief Python wrapper code for FSI computation by coupling pyBeam and SU2.
#  \author David Thomas, Rocco Bombardieri, Ruben Sanchez
#  \version 7.0.0
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



#############################    run with    mpirun -n 64 python3 pyAugustoFSI_Struct_Sens_FD.py -f config_primal.cfg






# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os, sys, shutil, copy
import time as timer

from optparse import OptionParser  # use a parser for configuration

from SU2_FSI.FSI_config import FSIConfig as io       # imports FSI config tools
from SU2_FSI import PrimalInterface as FSI # imports FSI python tools
import pyAugustoInterface as pyAugustoInterface
import pyMLSInterface as Spline_Module

# imports the CFD (SU2) module for FSI computation
import pysu2
import pyAugusto


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------

def Sens(options, dvID, perturbed_DV):

    if options.serial:
        comm = 0
        myid = 0
        numberPart = 1
        have_MPI = False
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
        numberPart = comm.Get_size()
        have_MPI = True

    rootProcess = 0

    # --- Set the working directory --- #
    if myid == rootProcess:
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
            print("Setting working directory : {}".format(os.getcwd()))
        else:
            print("Working directory is set to {}".format(os.getcwd()))

    # starts timer
    start = timer.time()

    confFile = str(options.filename)

    FSI_config = io(confFile)  # FSI configuration file
    CFD_ConFile = FSI_config['SU2_CONFIG']  # CFD configuration file
    AUG_ConFile = FSI_config['AUGUSTO_CONFIG']  # AUGUSTO  configuration file
    MLS_confFile = FSI_config['MLS_CONFIG_FILE_NAME']  # MLS configuration file
    INTERF_file = FSI_config['INTERFACE_NODES_FILE'] 

    if have_MPI:
        comm.barrier()


    # --- Initialize the fluid solver: SU2 --- #
    if myid == rootProcess:
        print('\n***************************** Initializing SU2 **************************************')
    try:
        FluidSolver = pysu2.CSinglezoneDriver(CFD_ConFile, 1, comm)
    except TypeError as exception:
        print('A TypeError occured in pysu2.CSingleZoneDriver : ', exception)
        if serial:
            print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please remove the --serial option that is incompatible with a parallel build.')
        else:
            print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, add --serial to launch your simulation.')
        return

    if have_MPI:
        comm.barrier()

    # --- Initialize the solid solver: pyBeam --- #
    if myid == rootProcess:
        print('\n***************************** Initializing pyBeam ************************************')
    try:

        SolidSolver = pyAugustoInterface.pyAugustoSolver(AUG_ConFile, INTERF_file)

        # update the design variable with the perturbed value
        SolidSolver.UpdateDesignVariable(dvID, perturbed_DV)
        

        print("---> P"+ str(myid) +": | SolidSolver.nPoint:" + str(SolidSolver.nPoint) + ": | SolidSolver.nPointLocal:" + str(SolidSolver.nPointLocal))
    except TypeError as exception:
            print('ERROR building the Solid Solver: ', exception)
    #else:
    #    SolidSolver = None

    if have_MPI:
        comm.barrier()

    # --- Initialize and set the coupling environment --- #
    if myid == rootProcess:
        print('\n***************************** Initializing FSI interface *****************************')
    try:
        FSIInterface = FSI.Interface(FSI_config, FluidSolver, SolidSolver, None, have_MPI)
    except TypeError as exception:
        print('ERROR building the FSI Interface: ', exception)

    if have_MPI:
        comm.barrier()


    if myid == rootProcess:
        print('\n***************************** Connect fluid and solid solvers *****************************')
    try:
        FSIInterface.connect(FSI_config, FluidSolver, SolidSolver)
    except TypeError as exception:
        print('ERROR building the Interpolation Interface: ', exception)

    if have_MPI:
        comm.barrier()

    if myid == rootProcess:  # we perform this calculation on the root core
        print('\n***************************** Initializing MLS Interpolation *************************')
    try:
            MLS = Spline_Module.pyMLSInterface(MLS_confFile, FSIInterface.globalFluidCoordinates, 
                                               FSIInterface.globalSolidCoordinates)
    except TypeError as exception:
            print('ERROR building the MLS Interpolation: ', exception)

    #else:
    #    MLS = None

    if have_MPI:
        comm.barrier()

    # Run the solver
    if myid == 0:
        print("\n------------------------------ Begin Solver -----------------------------\n")
    sys.stdout.flush()
    if have_MPI:
        comm.Barrier()

    cl, cd = FSIInterface.SteadyFSI(FSI_config, FluidSolver, SolidSolver, MLS, None)
    
    Js = SolidSolver.GetObjFunction() 

    if myid == rootProcess:
       print('DRAG COEFFICIENT: ', cd)

    # Postprocess the solver and exit cleanly
    FluidSolver.Postprocessing()

    if FluidSolver is not None:
        del FluidSolver


    return cd, Js


# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------
def main():
   # --- Get the FSI config file name form the command line options --- #
   parser = OptionParser()
   parser.add_option("-f", "--file", dest="filename",
                      help="read config from FILE", metavar="FILE")
   parser.add_option("--serial", action="store_true",
                      help="Specify if we need to initialize MPI", dest="serial", default=False)

   (options, args) = parser.parse_args()


   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   myid = comm.Get_rank()

   # --- This is only accessed if running from command prompt --- #
   delta = [5.00000000e-02, 1.23019153e-02, 3.02674238e-03, 7.44694566e-04,
 1.83223389e-04, 4.50799721e-05, 1.10913999e-05, 2.72890924e-06,
 6.71416206e-07, 1.65194105e-07, 4.06440777e-08, 1.00000000e-08]

   DV_ids = 1
   DV_values = 589500000000.0
   
   file = open("Sensitivity_FD_node_DV_" + str(DV_ids) + "_centered.txt", "w")
   file.write("DV id = {}\n".format(DV_ids))
   file.write("DV value = {}\n".format(DV_values))
   file.write("\n")
   file.write("\n")
   file.write("\n")
   file.flush()



   for i in range(len(delta)):

      delta_used = delta[i]

      # --- Set Parameter for surface sensitivity --- #
   
      file.write("Delta used = {}\n".format(delta_used))
      file.write("\n") 
      file.flush()

      perturbed_DV_plus = (1.0 + delta_used) * DV_values
      file.write("DV_plus = {:16.12f}\n".format(perturbed_DV_plus))
      drag_plus, Js_plus = Sens(options, DV_ids, perturbed_DV_plus)
      file.write("drag_plus = {:16.12f}\n".format(drag_plus))
      file.write("Js_plus = {:16.12f}\n".format(Js_plus))
      file.flush()

      if myid == 0:
         os.rename("historyFSI.dat", "historyFSI_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_plus.dat")
      

      perturbed_DV_minus = (1.0 - delta_used) * DV_values
      file.write("DV_minus = {:16.12f}\n".format(perturbed_DV_minus))
      drag_minus, Js_minus = Sens(options, DV_ids, perturbed_DV_minus)   
      file.write("drag_minus = {:16.12f}\n".format(drag_minus))
      file.write("Js_minus = {:16.12f}\n".format(Js_minus))
      file.flush()
      
      if myid == 0:
         os.rename("historyFSI.dat", "historyFSI_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_minus.dat")
      
      Cd_sens = (drag_plus - drag_minus) / (2 * delta_used * DV_values)
      Js_sens = (Js_plus - Js_minus) / (2 * delta_used * DV_values)

      file.write("Sensitivity Cd = {:25.22f}\n".format(Cd_sens))
      file.write("Sensitivity Js = {:25.22f}\n".format(Js_sens))
      file.flush()
      file.write("\n")
      file.write("\n")
      file.flush()
   
   file.close()

   return

# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------

# --- This is only accessed if running from command prompt --- #
if __name__ == '__main__':
    main()