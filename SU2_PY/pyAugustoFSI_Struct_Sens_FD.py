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




#############################    run with    mpirun -n 64 python3 pyAugustoFSI_Struct_Sens_FD.py -f config_primal.cfg -p /path/to/testcase -r crm.cfg crm_modal.cfg

#############################    -r lists one or more AUGUSTO config files (each with its own SMDAO
#############################    file) whose response is evaluated on the FSI-converged structural
#############################    state, for every DV perturbation. Include the FSI config itself
#############################    (AUGUSTO_CONFIG_FSI, e.g. crm.cfg) in -r if its own response should
#############################    be evaluated too -- it is no longer evaluated automatically.

#############################    SMDAO_TYPE must be RESPONSE in the FSI config and in every -r config
#############################    (checked upfront, before any analysis starts).

#############################    -p points at the folder with all the files needed for the analysis
#############################    (config_primal.cfg, AUGUSTO/MLS configs, mesh, interface file, the
#############################    -r configs and their own mesh/SMDAO files, etc.), same convention as
#############################    FSIshape_optimization.py. Its contents are copied flat into this
#############################    script's own folder (SU2_PY) and the analysis runs there, generating
#############################    whatever clutter (vtk, solid_load, restart, etc.) the solver produces.
#############################    At the end, a fresh FD_FSI_analysis folder is created containing a
#############################    clean 'testcase' copy of the -p inputs, plus the history, restart and
#############################    summary output files.





# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import os, sys, shutil, copy
import time as timer

import argparse

from SU2_FSI.FSI_config import FSIConfig as io       # imports FSI config tools
from SU2_FSI import PrimalInterface as FSI # imports FSI python tools
from SU2_FSI.FSI_tools import run_command, readConfig
import pyAugustoInterface as pyAugustoInterface
import pyMLSInterface as Spline_Module
from augusto_functions_toolbox import CheckSMDAOtype

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
    AUG_ConFile = FSI_config['AUGUSTO_CONFIG_FSI']  # AUGUSTO  configuration file
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
        if options.serial:
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

        SolidSolver = pyAugustoInterface.pyAugustoSolver(AUG_ConFile, INTERF_file, comm)

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

    # --- Evaluate every requested response on the FSI-converged structural
    # state. Each response gets its own standalone CAugusto instance, fed the
    # just-converged restart.pyAugusto (copied to the hardcoded "solution.pyAugusto"
    # that ReadPrimalSolution() expects) and the same perturbed DV as the coupled
    # run, then evaluated via EvalConstraintFSI() -- the same mechanism
    # struct_project.py uses for structural constraints/objective. This is
    # uniform for every response, whether it's the FSI config itself or a
    # separate one (e.g. modal), and safe to reuse "solution.pyAugusto" across
    # perturbations since everything here runs strictly sequentially: it is
    # overwritten only once the previous perturbation's responses are fully
    # evaluated. --- #
    responses = {}
    if options.responses:

        resp_solver = pyAugusto.CAugusto()
        resp_solver.InitialiseMPI(comm, True)

        for resp_cfg in options.responses:

            if myid == rootProcess:
                run_command('cp restart.pyAugusto solution.pyAugusto',
                            'Pulling FSI restart for ' + resp_cfg, False)
            if have_MPI:
                comm.barrier()

            resp_solver.Initialise(resp_cfg, os.getcwd() + '/')
            resp_solver.UpdateDesignVariable(dvID, perturbed_DV)
            resp_solver.EvalConstraintFSI()
            responses[resp_cfg] = resp_solver.GetObjFunction()
            resp_solver.Finalise()

            if have_MPI:
                comm.barrier()

    # Postprocess the solver and exit cleanly
    FluidSolver.Postprocessing()

    if FluidSolver is not None:
        del FluidSolver

    if SolidSolver is not None:
        del SolidSolver

    if FSIInterface is not None:
        del FSIInterface

    if MLS is not None:
       del MLS

    if have_MPI:
       comm.barrier()

    return cd, responses


# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------
def main():
   # --- Get the FSI config file name form the command line options --- #
   parser = argparse.ArgumentParser()
   parser.add_argument("-f", "--file", dest="filename", required=True,
                      help="read config from FILE", metavar="FILE")
   parser.add_argument("-p", "--path", dest="path", required=True,
                      help="path to the folder containing all the files needed for the analysis "
                           "(config_primal.cfg, AUGUSTO/MLS configs, mesh, interface file, the -r "
                           "configs and their own mesh/SMDAO files, etc.). Its contents are copied "
                           "into this script's folder to run, and a clean copy plus the results is "
                           "collected into FD_FSI_analysis at the end.",
                      metavar="PATH")
   parser.add_argument("-r", "--responses", dest="responses", nargs="*", default=[],
                      help="AUGUSTO config files (must already exist inside -p, each with its own "
                           "SMDAO file) whose response is evaluated on the FSI-converged structural "
                           "state, for every DV perturbation. Include the FSI config itself "
                           "(AUGUSTO_CONFIG_FSI) here too if its own response should be evaluated -- "
                           "it is not evaluated automatically anymore.",
                      metavar="CONFIG")
   parser.add_argument("--serial", action="store_true",
                      help="Specify if we need to initialize MPI", dest="serial", default=False)

   options = parser.parse_args()

   source_folder = os.path.abspath(options.path)

   # --- Resolve and validate every AUGUSTO config whose SMDAO response will
   # be evaluated (the FSI config itself, plus everything after -r): each
   # must exist in -p and have SMDAO_TYPE = RESPONSE. Done upfront, against
   # the original -p folder, before any staging or analysis starts. A config
   # listed both as the FSI config and in -r is simply checked twice. --- #
   primal_cfg_path = os.path.join(source_folder, options.filename)
   if not os.path.isfile(primal_cfg_path):
      parser.error('Primal config not found in ' + source_folder + ': ' + options.filename)

   fsi_augusto_cfg = readConfig(primal_cfg_path, 'AUGUSTO_CONFIG_FSI')

   for cfg_name in [fsi_augusto_cfg] + options.responses:
      cfg_path = os.path.join(source_folder, cfg_name)
      if not os.path.isfile(cfg_path):
         parser.error('AUGUSTO config not found in ' + source_folder + ': ' + cfg_name)
      CheckSMDAOtype(cfg_path, "RESPONSE")

   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   myid = comm.Get_rank()

   # Stage every analysis input file from -p directly into the current
   # working directory (SU2_PY), so the analysis runs here exactly as it did
   # before -p existed. Everything the solver generates along the way (vtk,
   # solid_load, restart files, etc.) is left here too. Only at the end are
   # the history/restart/summary outputs collected into FD_FSI_analysis,
   # alongside a clean copy of the testcase inputs.
   root_folder = os.getcwd()

   if myid == 0:
      run_command('cp -r ' + source_folder + '/. ' + root_folder + '/',
                  'Pulling testcase files from ' + source_folder, False)

   comm.barrier()

   delta = [0.01, 0.001, 0.0001, 0.00001]
   
   DV_ids = 20
   DV_values = 0.02 

   results = []
   summary_filename = "Sensitivity_FD_node_DV_" + str(DV_ids) + "_centered.txt"
   history_filenames = []
   restart_filenames = []

   # column width for each response's derivative column, wide enough for its label
   response_col_width = {r: max(25, len('d(' + r + ')/dDV') + 2) for r in options.responses}

   if myid == 0:
      outfile = open(summary_filename, "w")
      outfile.write("=" * 80 + "\n")
      outfile.write("  CENTERED FINITE DIFFERENCE SENSITIVITY ANALYSIS\n")
      outfile.write("=" * 80 + "\n")
      outfile.write("  DV id    = {}\n".format(DV_ids))
      outfile.write("  DV value = {:16.12f}\n".format(DV_values))
      outfile.write("  Responses evaluated = {}\n".format(
                    ", ".join(options.responses) if options.responses else "(none)"))
      outfile.write("=" * 80 + "\n\n")
      outfile.flush()

   for i in range(len(delta)):

      delta_used = delta[i]

      if myid == 0:
         outfile.write("-" * 80 + "\n")
         outfile.write("  Step {:d}/{:d} | delta = {:12.8e}\n".format(
                        i + 1, len(delta), delta_used))
         outfile.write("-" * 80 + "\n")
         outfile.flush()

      # --- Plus perturbation ---
      perturbed_DV_plus = (1.0 + delta_used) * DV_values
      if myid == 0:
         outfile.write("  DV+  = {:16.12f}\n".format(perturbed_DV_plus))
         outfile.flush()

      drag_plus, responses_plus = Sens(options, DV_ids, perturbed_DV_plus)

      if myid == 0:
         outfile.write("  Cd+  = {:16.12f}\n".format(drag_plus))
         for r in options.responses:
            outfile.write("  R[{}]+  = {:16.12f}\n".format(r, responses_plus[r]))
         outfile.write("\n\n")
         outfile.flush()

         plus_filename = "historyFSI_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_plus.dat"
         os.rename("historyFSI.dat", plus_filename)
         history_filenames.append(plus_filename)

         if options.responses:
            restart_plus_filename = "restart_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_plus.pyAugusto"
            os.rename("restart.pyAugusto", restart_plus_filename)
            restart_filenames.append(restart_plus_filename)

      # --- Minus perturbation ---
      perturbed_DV_minus = (1.0 - delta_used) * DV_values
      if myid == 0:
         outfile.write("  DV-  = {:16.12f}\n".format(perturbed_DV_minus))
         outfile.flush()

      drag_minus, responses_minus = Sens(options, DV_ids, perturbed_DV_minus)

      if myid == 0:
         outfile.write("  Cd-  = {:16.12f}\n".format(drag_minus))
         for r in options.responses:
            outfile.write("  R[{}]-  = {:16.12f}\n".format(r, responses_minus[r]))
         outfile.write("\n\n")
         outfile.flush()

         minus_filename = "historyFSI_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_minus.dat"
         os.rename("historyFSI.dat", minus_filename)
         history_filenames.append(minus_filename)

         if options.responses:
            restart_minus_filename = "restart_DV_" + str(DV_ids) + "_DH_" + str(delta_used) + "_minus.pyAugusto"
            os.rename("restart.pyAugusto", restart_minus_filename)
            restart_filenames.append(restart_minus_filename)

      # --- Sensitivities ---
      Cd_sens = (drag_plus - drag_minus) / (2 * delta_used * DV_values)
      response_sens = {r: (responses_plus[r] - responses_minus[r]) / (2 * delta_used * DV_values)
                        for r in options.responses}

      if myid == 0:
         outfile.write("  dCd/dDV = {:25.22f}\n".format(Cd_sens))
         for r in options.responses:
            outfile.write("  d({})/dDV = {:25.22e}\n".format(r, response_sens[r]))
         outfile.write("\n")
         outfile.flush()

      results.append((delta_used, drag_plus, drag_minus, Cd_sens, response_sens))

   if myid == 0:
      # --- Summary table (one column per response, labelled by its config file) --- #
      outfile.write("\n")
      outfile.write("=" * 80 + "\n")
      outfile.write("  SUMMARY\n")
      outfile.write("=" * 80 + "\n")

      header = "  {:>12s}  {:>16s}  {:>16s}  {:>25s}".format("delta", "Cd+", "Cd-", "dCd/dDV")
      for r in options.responses:
         header += "  {:>{w}s}".format("d(" + r + ")/dDV", w=response_col_width[r])
      outfile.write(header + "\n")
      outfile.write("  " + "-" * (len(header) - 2) + "\n")

      for (delta_used, cd_p, cd_m, cd_s, response_sens) in results:
         row = "  {:12.8e}  {:16.12f}  {:16.12f}  {:25.22f}".format(delta_used, cd_p, cd_m, cd_s)
         for r in options.responses:
            row += "  {:{w}.16e}".format(response_sens[r], w=response_col_width[r])
         outfile.write(row + "\n")

      outfile.write("=" * 80 + "\n")
      outfile.close()

      # --- Collect final results into FD_FSI_analysis --- #
      fd_folder = os.path.join(root_folder, 'FD_FSI_analysis')
      testcase_folder = os.path.join(fd_folder, 'testcase')

      if os.path.isdir(fd_folder):
         run_command('rm -r ' + fd_folder, 'Remove old FD_FSI_analysis folder', False)
      run_command('mkdir ' + fd_folder, 'Create FD_FSI_analysis folder', False)
      run_command('mkdir ' + testcase_folder, 'Create testcase folder', False)
      run_command('cp -r ' + source_folder + '/. ' + testcase_folder + '/',
                  'Copying clean testcase files into FD_FSI_analysis', False)

      for fname in [summary_filename] + history_filenames + restart_filenames:
         run_command('mv ' + os.path.join(root_folder, fname) + ' ' + os.path.join(fd_folder, fname),
                     'Moving ' + fname + ' into FD_FSI_analysis', False)

   return
# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------

# --- This is only accessed if running from command prompt --- #
if __name__ == '__main__':
    main()
