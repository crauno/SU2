#!/usr/bin/env python

## \file FSI_project.py
#  \brief FSI Shape Optimization project orchestrator.
#  \author Rocco Bombardieri based on work of  T. Lukaczyk, F. Palacios
#  \version 7.0.2 "Blackbird"
#
# SU2 Project Website: https://su2code.github.io
#
# The SU2 Project is maintained by the SU2 Foundation
# (http://su2foundation.org)
#
# Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
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

import copy
import numpy as np
from math import pow, factorial, pi
import time, os, sys
from SU2_FSI.FSI_config import FSIConfig as FSIConfig
from SU2_FSI import FSI_design
from SU2_FSI.FSI_tools import run_command, readConfig, MakeDir, CopyFile, UpdateConfig, PullingPrimalAdjointFiles, PullRestartFiles, readDVParam, ReadPointInversion, WriteSolution, Fix_FFD_CP, ReadSensAoA, ReadPrimalCD
from SU2_FSI.FSI_design import Design
from SU2_FSI.FSI_tools import  readConfig
from structopt.pystructopt.pyoptlib.struct_config import OptConfig as StructOptConfig
from structopt.pystructopt.pyoptlib.struct_project import Project as StructProject
from structopt.pystructopt.pyoptlib.struct_tools import readConfig as readAUGUSTOConfig
# -------------------------------------------------------------------
#  Project Class
# -------------------------------------------------------------------

class Project:
    
    """ 
    Project interface class that handles FSI shape optimization
    """

    def __init__(self, config ):
        """
        Class constructor. Declare some variables and do some screen outputs.
        """
        print('Initializing project....')
        self.config = config  # FSI optimization config object
        
        folder = self.config['FOLDER']  # root folder where optimization is done
        #folder = folder.rstrip('/')
        self.folder  = folder
        self.source_folder = self.config['SOURCE_FOLDER']  # folder (-p) holding all the analysis input files
        self.design_folder = ''
        self.deform_folder = ''
        self.geo_folder = ''
        self.primal_folder = ''
        self.adjoint_folder = ''
        self.opt_mode = self.config['OPT_MODE']

        self.design_toll = 10**(-20)  # allowable difference into design variable vector to consider the same design

        # config objects for primal and adjoint simulations with structural and fluid config files and options
        self.configFSIPrimal = None
        self.configFSIAdjoint = None

        # structural project. Initialised in case structrual DVs or objective are present
        self.structProject = None

        # Design container
        self._design = []

        self.design_iter = -1  # optimization iter (design number) [initialization]
        self.magnord_design = 3 # Expected order of magnitude of design number

        # clean previous designs
        self.clean_previous_designs()

        # Creating design folder
        self.create_design_folder()

        # Stage every analysis input file from self.source_folder (-p) into
        # DESIGNS/testcase. From here on, self.testcase_folder is the reference
        # folder the solver pulls files from, replacing SU2_PY's former role.
        self.create_testcase_folder()
        self.config['TESTCASE_FOLDER'] = self.testcase_folder

        # Design variable options
        # Reading point inversion
        config_def = self.testcase_folder + '/' + self.config['CONFIG_DEF']
        MeshFile = readConfig(config_def, 'MESH_FILENAME')
        self.PointInv = ReadPointInversion(config_def, self.testcase_folder + '/' + MeshFile)

        # reading Bezier curves order
        line = readConfig(config_def, 'FFD_DEGREE')
        line = line.strip('( )')
        line = line.split(',')
        self.ffd_degree = [int(float(line[0])),int(float(line[1])),int(float(line[2]))]

        # DV_values indexes (control points indexes of the FD Box)
        self.FFD_indexes = readDVParam(config_def)
        self.n_dv = 0 # number of design variables

        # check if to fix the CP  on the root of the wing
        if self.config['FFD_CONSTRAINT'] == 'ROOT':
           self.ffd_fixed = Fix_FFD_CP(self.ffd_degree)

        # project memorizes config of both adjoin and primal solvers (top level config)
        self.setup_configs()


    def setup_configs(self):
        '''
        Reads config files for primal and Adjoint simulations
        '''
        # Read primal config
        self.configFSIPrimal = FSIConfig(self.testcase_folder + '/' + self.config['CONFIG_PRIMAL'])
        # Read Adjoint config
        self.configFSIAdjoint = FSIConfig(self.testcase_folder + '/' + self.config['CONFIG_ADJOINT'])
        # Locate AUGUSTO meshfile
        self.pyAugustoMesh = readConfig(self.testcase_folder + '/' + self.configFSIPrimal['AUGUSTO_CONFIG_FSI'], 'INPUT_FILENAME')
        # Locate AUGUSTO smdao
        self.pyAugustoSmdao = readConfig(self.testcase_folder + '/' + self.configFSIPrimal['AUGUSTO_CONFIG_FSI'], 'SMDAO_FILENAME')
        # Locate file with interface nodes
        self.pyInterfaceFile = readConfig(self.testcase_folder + '/' + self.config['CONFIG_PRIMAL'], 'INTERFACE_NODES_FILE')

    
    def InitStructProject(self):

        if self.opt_mode == "STRUCT":

           print('Initializing structural project...')

           structConfig = StructOptConfig(self.folder, self.config['CONFIG_STRUCT_OPT'], self.configFSIPrimal['AUGUSTO_CONFIG_FSI'], self.testcase_folder, self.config['NUMBER_PART'])
           
           self.structProject = StructProject(structConfig, False)

           self.structProject.InitNormalizedVariables()


    def CheckOptCase(self):

        """
        Identifies which of the optimisation cases is being run (based on
        OPT_MODE) and validates the configuration for that case.

        OBJECTIVE_WEIGHT is not checked here: it is written directly by the
        Python orchestrator into each per-role copy of the shared adjoint
        config at runtime (0.0 for the numerator run, 1.0 for the denominator
        run, W for the fixed-CL-corrected run), so whatever static value sits
        in the template is unconditionally overwritten before every analysis
        and carries nothing worth validating upfront.
        """

        if self.opt_mode not in ["AERO", "STRUCT"]:

            sys.exit(self.opt_mode + " is an invalid option for OPT_MODE field. Available options are either AERO or STRUCT. ")

        print('==============================================================')
        print('  Optimisation case summary')
        print('==============================================================')

        if self.opt_mode == "AERO":
            print('Optimisation case: AERO objective / AERO design variables')

        elif self.opt_mode == "STRUCT":
            print('Optimisation case: STRUCT objective / STRUCT design variables')

            adjoint_flow_config = self.testcase_folder + '/' + self.configFSIAdjoint['SU2_CONFIG']

            # FIXED_CL_MODE: only the adjoint config must be NO -- the fixed-CL correction is
            # applied by the python orchestrator, not by SU2's own trim driver. The primal is
            # free to be YES (a real trimmed design point) or NO (a fixed-AoA case).
            fixed_cl_mode = readConfig(adjoint_flow_config, 'FIXED_CL_MODE', False)
            if fixed_cl_mode != 'NO':
                sys.exit('FIXED_CL_MODE must be NO in ' + adjoint_flow_config +
                          ' when OPT_MODE = STRUCT (the fixed-CL correction is applied by the'
                          ' python orchestrator, not by SU2 s own trim driver). Found: ' + fixed_cl_mode)

            # OBJECTIVE_FUNCTION must be LIFT so the shared adjoint template is ready for the
            # python orchestrator to seed OBJECTIVE_WEIGHT per-role at runtime. Not checked on
            # the primal config: OBJECTIVE_FUNCTION is never read during MATH_PROBLEM=DIRECT
            # (Evaluate_ObjFunc is only called from the discrete-adjoint drivers), so it's inert there.
            objective_function = readConfig(adjoint_flow_config, 'OBJECTIVE_FUNCTION', False)
            if objective_function != 'LIFT':
                sys.exit('OBJECTIVE_FUNCTION must be LIFT in ' + adjoint_flow_config +
                          ' when OPT_MODE = STRUCT. Found: ' + objective_function)


    def obj_f(self,dvs):
      
        print('Calling aero obj_f') 
        #x_in = copy.deepcopy(dvs)
        # Checking if new design is needed
        # In case starts new design and deform
        #self.CheckNewDesign(x_in)
                 
        #Primal
        self.Primal()
        
        # pulling obj function with scale
        obj_f, scale, global_factor = self._design[self.design_iter].pull_obj_f(self.primal_folder)
        # return function
        return obj_f*scale*global_factor
        
    def obj_df(self,dvs):

        print('Calling aero obj_df')   
        #x_in = copy.deepcopy(dvs)
        # Check if new design is needed (it won't as Adjoin is performed after primal)        
        # In case start new design and deform
        #self.CheckNewDesign(x_in)
        
        #Adjoint
        self.Adjoint()
        
        # return the function
        obj_df, global_factor = self._design[self.design_iter].pull_obj_df(self.adjoint_folder,self.FFD_indexes, self.PointInv,self.ffd_degree)
                
        # check if the root has to be fixed
        if self.config['FFD_CONSTRAINT'] == 'ROOT':
           obj_df = self.Fix_FFD_CP_grads(obj_df,'OF')
                
        return obj_df*global_factor

    def con_ceq(self,dvs):
      
        print('Calling aero con_ceq')
        x_in = copy.deepcopy(dvs)
        # Check if new design is needed        
        # In case start new design and deform
        self.CheckNewDesign(x_in)
        
        
        #Check if Geo has been executed, if it hasn't, execute Geo
        self.CheckGeo()
        
        # pulls constraint equality
        c_eq, global_factor = self._design[self.design_iter].pull_c_eq(self.geo_folder)
        
        # return ceq        
        return c_eq* global_factor
    
    def con_dceq(self,dvs):
      
        print('Calling aero con_dceq')
        x_in = copy.deepcopy(dvs)
        # Check if new design is needed (it won't as geo gradient is calculated after geo)       
        # In case start new design and deform
        self.CheckNewDesign(x_in)

        #Check if Geo has been executed, if it hasn't, execute Geo
        self.CheckGeo()   
        
        # pull gradient of constraint equality
        dc_eq, global_factor = self._design[self.design_iter].pull_c_deq( self.geo_folder)
        
        # check if the root has to be fixed
        if self.config['FFD_CONSTRAINT'] == 'ROOT':
           dc_eq = self.Fix_FFD_CP_grads(dc_eq,'CONSTR')        
        
        # return dceq
        return dc_eq*global_factor
    
    def con_cieq(self,dvs):
      
        print('Calling aero con_cieq')
        #x_in = copy.deepcopy(dvs)
        # Check if new design is needed        
        # In case start new design and deform
        #self.CheckNewDesign(x_in)
        
        #Check if Geo has been executed, if it hasn't, execute Geo
        self.CheckGeo()
        
        # pull constraint inequality
        c_ieq, global_factor = self._design[self.design_iter].pull_c_ieq(self.geo_folder)
        
        # return cieq
        return c_ieq* global_factor
    
    def con_dcieq(self,dvs):
      
        print('Calling aero con_dcieq')
        #x_in = copy.deepcopy(dvs)
        # Check if new design is needed (it won't as geo gradient is calculated after geo)       
        # In case start new design and deform
        #self.CheckNewDesign(x_in)

        #Check if Geo has been executed, if it hasn't, execute Geo
        self.CheckGeo()
        
        # pull gradient of constraint inequality
        c_dieq, global_factor = self._design[self.design_iter].pull_c_dieq(self.geo_folder)

        # check if the root has to be fixed
        if self.config['FFD_CONSTRAINT'] == 'ROOT':
           c_dieq = self.Fix_FFD_CP_grads(c_dieq,'CONSTR')         
       
        # return dcieq    
        return c_dieq*global_factor
        
        
    def clean_previous_designs(self):

        # Removing old designs
        t0 = time.time()
        print('Removing old designs in 5 sec...')
        elapsed = 0
        while elapsed <=5:
           t = time.time() 
           elapsed = t - t0
        print('Done!')   
        command = 'rm -r ' + self.folder + '/DESIGNS'
        
        # Executes shell command
        run_command(command, 'Remove old designs', False)     
    
    
    def create_design_folder(self):

        MakeDir(self.folder + '/DESIGNS', 'Create design folder')


    def create_testcase_folder(self):

        self.testcase_folder = self.folder + '/DESIGNS/testcase'

        MakeDir(self.testcase_folder, 'Create testcase folder')

        # Stage all the analysis input files (config, mesh, AUGUSTO configs, etc.)
        command = 'cp -r ' + self.source_folder + '/. ' + self.testcase_folder + '/'
        run_command(command, 'Pulling testcase files from ' + self.source_folder, False)



    def CheckNewDesign(self, x_in):
       
       # if the optimisation is with structural DVs and objective, new designs are initialised
       # by the structrual project
       if self.opt_mode == "STRUCT" :
          return

       if self.design_iter == -1:
           print('Evaluating initial design')
           self.design_iter = self.design_iter + 1
           # starting new design
           self.InitializeNewDesign(x_in)
           # Writing solution to Output           
           WriteSolution(self.folder + '/DESIGNS' ,x_in,self.design_iter)
       else:            
          x =  self._design[self.design_iter].getdv()   
          delta = x - x_in
          module = np.linalg.norm(delta)
          if module > self.design_toll:
             print('Evaluating new design')
             self.design_iter = self.design_iter + 1
             # starting new design
             self.InitializeNewDesign(x_in)
             # Writing solution to Output
             WriteSolution(self.folder + '/DESIGNS' ,x_in,self.design_iter)
             # performing mesh deform
             self.DeformMesh()
          else:
             print('Using previous design')
            
    
    def InitializeNewDesign(self,x_in):  
        # old design
        if self.design_iter == 0:
           x_old = x_in
        else:   
           x_old = self._design[self.design_iter-1].getdv()
        
        # create design folder
        self.design_folder = self.folder + '/DESIGNS/' + 'DSN_'+ str(int(self.design_iter)).zfill(self.magnord_design)
        MakeDir(self.design_folder, 'Creating design ' + str(int(self.design_iter)).zfill(self.magnord_design) + ' directory')
            
        print('InitializeNew Design x_in = {}'.format(x_in))    
        # initialize and append new design object    
        self._design.append(Design(self.config,self.configFSIPrimal,self.configFSIAdjoint, self.testcase_folder, self.design_folder, self.design_iter ,x_in, x_old ))
        

    def DeformMesh(self):    
        
        # old design
        #x = self.design[self.design_iter].x
        
        # Check if there is the need to deform the mesh
        # It is FALSE if any x of the current design is different than 0
        #all_zeros = not np.any( x )
        
        #if all_zeros == False:
            
        # create folder for analysis
        self.deform_folder = self.design_folder + '/DEFORM'
        MakeDir(self.deform_folder, 'Creating deform directory for design ' + str(int(self.design_iter)).zfill(self.magnord_design))
           
        # pull config deformation file
        config_deform = self.testcase_folder + '/' + self.config['CONFIG_DEF']
        CopyFile(config_deform, self.deform_folder + '/', 'Pulling deformation config')

        # creating a symbolic link to original meshfile
        mesh_filename = readConfig(config_deform, 'MESH_FILENAME')
        command = 'ln -s ' + self.testcase_folder + '/' + mesh_filename + ' ' + self.deform_folder + '/' + mesh_filename
        run_command(command, 'Pulling mesh config for deformation', False)
        
        # Performing mesh deformation
        self._design[self.design_iter].SU2_DEF(self.deform_folder)
        
        
    def CheckGeo(self):    
       """
       If Geo sensitivities (constraints and gradients). are required, checks if GEO run has been done. If not sets up the folder and performs it.
       """ 
       
       if self._design[self.design_iter].geo == False:
            
           # creating folder for analysis
           self.geo_folder = self.design_folder + '/GEO'
           MakeDir(self.geo_folder, 'Creating GEO directory for design ' + str(int(self.design_iter)).zfill(self.magnord_design))
           
           # pulling geo deformation file
           config_geo = self.testcase_folder + '/' + self.config['CONFIG_GEO']
           CopyFile(config_geo, self.geo_folder + '/', 'Pulling geo config')
           # pulling mesh file 
           self.SetMesh(self.geo_folder)
           
           # Running SU2_GEO
           self._design[self.design_iter].SU2_GEO(self.geo_folder)
           
    
    def Primal(self):
       """
       Sets up and Performs primal solver
       """
       if self._design[self.design_iter].primal == False:

           # creating folder for analysis
           self.primal_folder = self.design_folder + '/Primal'
           MakeDir(self.primal_folder, 'Creating Primal directory for design ' + str(int(self.design_iter)).zfill(self.magnord_design))
            
           PullingPrimalAdjointFiles(self.testcase_folder, self.primal_folder, self.configFSIPrimal, self.configFSIPrimal['AUGUSTO_CONFIG_FSI'], self.pyInterfaceFile)
           # pulling mesh file 
           self.SetMesh(self.primal_folder)  
           
           # Running primal
           self._design[self.design_iter].FSIPrimal(self.primal_folder)
       
       else:
           print("Primal FSI problem already solved: only pulling the result")
       
    def Adjoint(self):
       """
       Sets up and Performs Adjoint solver
       """          
       
       # creating folder for analysis
       self.adjoint_folder = self.design_folder + '/Adjoint'
       MakeDir(self.adjoint_folder, 'Creating Adjoint directory for design ' + str(int(self.design_iter)).zfill(self.magnord_design))
       
       PullingPrimalAdjointFiles(self.testcase_folder, self.adjoint_folder, self.configFSIAdjoint, self.configFSIAdjoint['AUGUSTO_CONFIG_FSI'], self.pyInterfaceFile)

       # pulling mesh file 
       self.SetMesh(self.adjoint_folder)         
       
       # pulling restart for pyBeam and SU2 and flow.vtk
       if self._design[self.design_iter].primal == True:

          PullRestartFiles(self.primal_folder, self.adjoint_folder)

       else:
          print('Primal not yet available, can t pull solutions for Adjoint....')
          sys.exit()

       # Running adjoint
       self._design[self.design_iter].FSIAdjoint(self.adjoint_folder, None)
            
    def SetMesh(self, destination_folder):
       """
       Pulls mesh file. If optimization iter is 1, pulling is from project folder. If a deformation occurred, pulling is done from DEFORM folder
       """ 
       
       if self._design[self.design_iter].deformation == False:
          # In case deformation hasn't occurred (first iteration) we need the original mesh file
          mesh_filename = readConfig(self.testcase_folder + '/' + self.config['CONFIG_DEF'], 'MESH_FILENAME')
          command = 'ln -s ' + self.testcase_folder + '/' + mesh_filename + ' ' + destination_folder + '/' + mesh_filename
          run_command(command, 'Pulling mesh config for deformation', False)

       else:
          # in case deformation has occurred mesh file is named as output of SU2_DEF and needs to be pulled from the dedicated folder
          mesh_filename = readConfig(self.testcase_folder + '/' + self.config['CONFIG_DEF'], 'MESH_OUT_FILENAME')
          command = 'ln -s ' + self.deform_folder + '/' + mesh_filename + ' ' + destination_folder + '/' + mesh_filename
          run_command(command, 'Pulling mesh config for deformation', False)
          
           
           
    def Fix_FFD_CP_grads(self,gradient,gradient_type):
        
        if gradient_type == 'OF':            
            for i in range(self.ffd_fixed.size):
                gradient[self.ffd_fixed[i]] = 0.0
        
        elif gradient_type == 'CONSTR':
            for i in range(self.ffd_fixed.size):
                gradient[:,self.ffd_fixed[i]] = 0.0
            
            
        return gradient





    def ConnectProjects(self):
        
        self.design_iter = self.structProject.design_iter
        self.design_folder = self.structProject.design_folder
        self.primal_folder = self.structProject.design_folder_primal
        self.adjoint_folder = self.structProject.design_folder_adjoint


        #print(self.design_iter)
        #print()
        #print(self.design_folder)
        #print()
        #print(self.primal_folder)
        #print()
        #print(self.adjoint_folder)
        #print()



        if self.structProject.initialised_new_design :

           self._design.append(Design(self.config,self.configFSIPrimal,self.configFSIAdjoint, self.testcase_folder, self.design_folder, self.design_iter , None, None ))
           
           self.structProject.initialised_new_design = False



    def CheckNewFSIDesign(self, x):
        
        x_in = copy.deepcopy(x)

        if self.opt_mode == "STRUCT" :
           
           self.structProject.CheckNewDesign(x_in, True)

           self.ConnectProjects()

        elif self.opt_mode == "AERO" :

           self.CheckNewDesign(x_in)







    def CheckStructuralSetup(self):

       """
       Make some checks on the configuration and mesh files regarding the structural optimisation
       """
       
       # check that the mesh file is unique     
       mesh_files = [self.pyAugustoMesh, self.structProject.pyAugustoMeshObjf] + self.structProject.pyAugustoMeshConstr

       if len(set(mesh_files)) != 1:

         sys.exit("\nError: Multiple FE meshfiles detected. In the current implementation, all the FE configuration files must point to a single mesh file")




    def ComputeLiftCoeffSensitivity(self):

       """
       Solves the shared dC_L/dAoA adjoint (the denominator L_A of
       W = sigma_A/L_A, used by every constraint's fixed-CL correction).
       No structural response is seeded here (-c NONE): the flow objective
       is seeded with OBJECTIVE_FUNCTION=LIFT, OBJECTIVE_WEIGHT=1.0.

       Returns (L_A, primal_AoA): L_A = dC_L/dAoA, corrected for the missing
       explicit rotation term in SU2's raw Sens_AoA (dCD/dAoA and dCL/dAoA are
       each missing the explicit d(rotation)/dAoA piece, worth +CL*(pi/180)
       for CD and -CD*(pi/180) for CL -- see su2_sens_aoa_rotation_bug
       memory). primal_AoA is the primal's trimmed AoA read from flow.meta
       (None if the primal ran FIXED_CL_MODE=NO, i.e. flow.meta has no AOA=
       line at all).
       """

       cl_sensitivity_folder = self.structProject.design_folder_adjoint + '/CL_sensitivity'

       MakeDir(cl_sensitivity_folder, 'Creating subdirectory for CL sensitivity')

       # pull files for analysis (includes the adjoint config itself); using the
       # base FSI AUGUSTO config since no structural response is seeded here
       PullingPrimalAdjointFiles(self.testcase_folder, cl_sensitivity_folder, self.configFSIAdjoint, self.configFSIAdjoint['AUGUSTO_CONFIG_FSI'], self.pyInterfaceFile)

       # pulling mesh file
       self.SetMesh(cl_sensitivity_folder)

       # pulling restart for pyAugusto and SU2 and flow.vtk
       if self._design[self.design_iter].primal == True:

          PullRestartFiles(self.primal_folder, cl_sensitivity_folder)

       else:
         print('Primal not yet available, can t pull solutions for Adjoint....')
         sys.exit()

       adj_config_file = cl_sensitivity_folder + '/' + self.configFSIAdjoint['SU2_CONFIG']

       # propagate the primal's trimmed AoA (guarded: flow.meta is only written
       # at all -- let alone with an AOA= line -- if the primal ran
       # FIXED_CL_MODE=YES; WriteAdditionalFiles gates the call to
       # WriteMetaData on config->GetFixed_CL_Mode(), CFlowOutput.cpp:950-953.
       # Nothing to propagate otherwise, since the adjoint's own static AOA
       # already matches the primal's by construction when neither one trims)
       flow_meta_file = cl_sensitivity_folder + '/flow.meta'
       primal_AoA = readConfig(flow_meta_file, 'AOA', False) if os.path.isfile(flow_meta_file) else 'NO'
       if primal_AoA == 'NO':
           primal_AoA = None
       else:
           UpdateConfig(adj_config_file, 'AOA', primal_AoA)
           primal_AoA = float(primal_AoA)

       # seed OBJECTIVE_WEIGHT = 1.0 (OBJECTIVE_FUNCTION = LIFT already enforced by CheckOptCase)
       UpdateConfig(adj_config_file, 'OBJECTIVE_WEIGHT', '1.0')

       # Running adjoint (-c NONE -> no structural response seeded)
       self._design[self.design_iter].FSIAdjoint(cl_sensitivity_folder, None)

       # raw dC_L/dAoA, missing the explicit rotation term
       dCl_dAoA_raw = ReadSensAoA(cl_sensitivity_folder + '/history.csv')

       # primal's converged CD, needed for the rotation-term correction
       CD_primal = ReadPrimalCD(self.primal_folder + '/historyFSI.dat')

       # corrected L_A = dC_L/dAoA
       dCl_dAoA = dCl_dAoA_raw - CD_primal * (pi / 180.0)

       return dCl_dAoA, primal_AoA


    def ComputeStructRespSensitivity_FixedAoA(self, i):

       """
       Solves constraint i's numerator adjoint (sigma_A,i = dJs_i/dAoA, and the
       fixed-AoA structural-DV gradient G_AoA,i), inside a FixedAoA_sensitivity
       subfolder of that constraint's own adjoint folder. The structural response
       is registered and seeded (-c AUGUSTO_CONFIG_CONSTR[i]); the flow objective
       contributes nothing (OBJECTIVE_WEIGHT=0.0), so it doesn't contaminate
       either quantity read out of this run.

       Assumes the constraint's own adjoint folder already exists (created by
       the caller). Returns sigma_A,i = dJs_i/dAoA (the numerator of W_i).
       """

       augusto_constr_cfg = self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i]

       # constraint's own adjoint folder (e.g. Adjoint/crm_stress), already created by the caller
       current_adj_folder = self.structProject.design_folder_adjoint + '/' + augusto_constr_cfg.split('.')[0]

       # nested subfolder for this specific (fixed-AoA) analysis
       fixed_aoa_folder = current_adj_folder + '/FixedAoA_sensitivity'
       MakeDir(fixed_aoa_folder, 'Creating subdirectory for fixed-AoA sensitivity of constraint ' + augusto_constr_cfg)

       # pull files for analysis (includes the adjoint config itself)
       PullingPrimalAdjointFiles(self.testcase_folder, fixed_aoa_folder, self.configFSIAdjoint, augusto_constr_cfg, self.pyInterfaceFile)

       # pulling mesh file
       self.SetMesh(fixed_aoa_folder)

       # pulling restart for pyAugusto and SU2 and flow.vtk
       if self._design[self.design_iter].primal == True:

          PullRestartFiles(self.primal_folder, fixed_aoa_folder)

       else:
         print('Primal not yet available, can t pull solutions for Adjoint....')
         sys.exit()

       adj_config_file = fixed_aoa_folder + '/' + self.configFSIAdjoint['SU2_CONFIG']

       # propagate the primal's trimmed AoA (guarded: flow.meta is only written
       # at all if the primal ran FIXED_CL_MODE=YES; nothing to do otherwise)
       flow_meta_file = fixed_aoa_folder + '/flow.meta'
       primal_AoA = readConfig(flow_meta_file, 'AOA', False) if os.path.isfile(flow_meta_file) else 'NO'
       if primal_AoA != 'NO':
           UpdateConfig(adj_config_file, 'AOA', primal_AoA)

       # seed OBJECTIVE_WEIGHT = 0.0 (flow objective contributes nothing; the
       # structural response, seeded below via -c, is the only thing differentiated)
       UpdateConfig(adj_config_file, 'OBJECTIVE_WEIGHT', '0.0')

       # Running adjoint (-c <constraint config> -> structural response seeded)
       self._design[self.design_iter].FSIAdjoint(fixed_aoa_folder, augusto_constr_cfg)

       # return dJs/dAoA
       dJs_dAoA = ReadSensAoA(fixed_aoa_folder + '/history.csv')

       return dJs_dAoA


    def ComputeStructRespSensitivity_FixedCl(self, i, W_i):

       """
       Solves constraint i's fixed-CL corrected adjoint -- the actual
       constraint gradient at fixed CL, G_CL,i. Runs directly in the
       constraint's own adjoint folder (already created by the caller).
       Seeds OBJECTIVE_FUNCTION=LIFT (already enforced by CheckOptCase) with
       OBJECTIVE_WEIGHT=-W_i, alongside the structural response (-c
       AUGUSTO_CONFIG_CONSTR[i]): because the discrete adjoint's RHS is linear
       in the seed, this makes the run converge to exactly G_CL,i.
       """

       augusto_constr_cfg = self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i]

       # constraint's own adjoint folder (e.g. Adjoint/crm_stress), already created by the caller
       current_adj_folder = self.structProject.design_folder_adjoint + '/' + augusto_constr_cfg.split('.')[0]

       # pull files for analysis (includes the adjoint config itself)
       PullingPrimalAdjointFiles(self.testcase_folder, current_adj_folder, self.configFSIAdjoint, augusto_constr_cfg, self.pyInterfaceFile)

       # pulling mesh file
       self.SetMesh(current_adj_folder)

       # pulling restart for pyAugusto and SU2 and flow.vtk
       if self._design[self.design_iter].primal == True:

          PullRestartFiles(self.primal_folder, current_adj_folder)

       else:
         print('Primal not yet available, can t pull solutions for Adjoint....')
         sys.exit()

       adj_config_file = current_adj_folder + '/' + self.configFSIAdjoint['SU2_CONFIG']

       # propagate the primal's trimmed AoA (guarded: flow.meta is only written
       # at all if the primal ran FIXED_CL_MODE=YES; nothing to do otherwise)
       flow_meta_file = current_adj_folder + '/flow.meta'
       primal_AoA = readConfig(flow_meta_file, 'AOA', False) if os.path.isfile(flow_meta_file) else 'NO'
       if primal_AoA != 'NO':
           UpdateConfig(adj_config_file, 'AOA', primal_AoA)

       # seed OBJECTIVE_WEIGHT = -W_i (OBJECTIVE_FUNCTION = LIFT already enforced by CheckOptCase)
       UpdateConfig(adj_config_file, 'OBJECTIVE_WEIGHT', str(-W_i))

       # Running adjoint (-c <constraint config> -> structural response seeded)
       self._design[self.design_iter].FSIAdjoint(current_adj_folder, augusto_constr_cfg)


    def con_struct_dcieq_normalized(self):

       """
       Solve the coupled adjoint problem for strutural optimisation constraint
       """

       # solve the shared CL-sensitivity adjoint needed by every constraint's W
       dCl_dAoA, primal_AoA = self.ComputeLiftCoeffSensitivity()

       # create adjoint constraint subfolders and pull the needed files
       self.structProject.constr_subfolders_adjoint = []
       dJsi_dAoA_list = []
       W_list = []

       for i in range(len(self.structProject.config['AUGUSTO_CONFIG_CONSTR'])):

            # create current adjoint constraint subfolder
            current_adj_folder = self.structProject.design_folder_adjoint + '/' + self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i].split('.')[0]
            self.structProject.constr_subfolders_adjoint.append(current_adj_folder)

            MakeDir(current_adj_folder, 'Creating subdirectory for constraint ' + self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i])

            # solve constraint i's fixed-AoA sensitivity (dJsi_dAoA, needed for W_i), in its own subfolder
            dJsi_dAoA = self.ComputeStructRespSensitivity_FixedAoA(i)

            # W_i = sigma_A,i / L_A
            W_i = dJsi_dAoA / dCl_dAoA
            dJsi_dAoA_list.append(dJsi_dAoA)
            W_list.append(W_i)

            # solve constraint i's fixed-CL corrected adjoint (the actual constraint gradient)
            self.ComputeStructRespSensitivity_FixedCl(i, W_i)

       # log dCl_dAoA and, per constraint, dJs_dAoA/W_i -- one file per design point
       summary_file = self.structProject.design_folder_adjoint + '/Sensitivity_FixedCL_summary.txt'
       summary = open(summary_file, 'w')
       summary.write('=' * 80 + '\n')
       summary.write('  FIXED-CL CORRECTION SUMMARY\n')
       summary.write('=' * 80 + '\n')
       summary.write('  dCl_dAoA (L_A, shared) = {:.10e}\n'.format(dCl_dAoA))
       if primal_AoA is not None:
           summary.write('  Primal trimmed AoA (from flow.meta) = {:.10f} deg\n'.format(primal_AoA))
       else:
           summary.write('  Primal trimmed AoA (from flow.meta) = N/A (primal FIXED_CL_MODE=NO)\n')
       summary.write('=' * 80 + '\n\n')
       summary.write('  {:<30s}  {:>20s}  {:>20s}\n'.format('Constraint', 'dJs_dAoA (sigma_A)', 'W_i'))
       summary.write('  ' + '-' * 74 + '\n')
       for i in range(len(self.structProject.config['AUGUSTO_CONFIG_CONSTR'])):
           summary.write('  {:<30s}  {:>20.10e}  {:>20.10e}\n'.format(
                         self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i], dJsi_dAoA_list[i], W_list[i]))
       summary.write('=' * 80 + '\n')
       summary.close()

       # pull non normalized gradient of constraint inequality
       c_dieq, global_factor = self.structProject._design[self.design_iter].pull_c_dieq(self.structProject.constr_subfolders_adjoint)
        
       dcon_physical =  np.array(c_dieq) * global_factor


       # Scale gradients for normalized space
       if dcon_physical.ndim == 1:
            dcon_normalized = dcon_physical * self.structProject.x_range
       else:
            # Multiple constraints - scale each column
            dcon_normalized = dcon_physical.copy()
            for i in range(len(self.structProject.x_range)):
                dcon_normalized[:, i] *= self.structProject.x_range[i]
        
       return dcon_normalized
              