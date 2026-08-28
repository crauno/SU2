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
from math import pow, factorial
import time, os, sys
from SU2_FSI.FSI_config import FSIConfig as FSIConfig
from SU2_FSI import FSI_design
from SU2_FSI.FSI_tools import run_command, readConfig, MakeDir, PullingPrimalAdjointFiles, PullRestartFiles, readDVParam, ReadPointInversion, WriteSolution, Fix_FFD_CP
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
        OPT_MODE and, for OPT_MODE = STRUCT, the SU2 adjoint OBJECTIVE_WEIGHT)
        and validates the configuration for that case.

        The objective scaling is handled entirely on the python side
        (obj_scale, global_factor), so OBJECTIVE_WEIGHT in the SU2 adjoint
        config is only ever allowed to be 1.0 (or absent/commented out, which
        defaults to 1.0) or 0.0:

        AERO   (aero objective, aero design variables): OBJECTIVE_WEIGHT must
               be 1.0 (or absent).

        STRUCT (structural design variables):
               - OBJECTIVE_WEIGHT = 1.0 (or absent) -> aero objective
                 differentiated with respect to structural design variables.
               - OBJECTIVE_WEIGHT = 0.0 -> structural objective
                 differentiated with respect to structural design variables.

        Any other OBJECTIVE_WEIGHT value raises an error and stops the
        optimisation.
        """

        if self.opt_mode not in ["AERO", "STRUCT"]:

            sys.exit(self.opt_mode + " is an invalid option for OPT_MODE field. Available options are either AERO or STRUCT. ")

        print('==============================================================')
        print('  Optimisation case summary')
        print('==============================================================')

        adjoint_flow_config = self.testcase_folder + '/' + self.configFSIAdjoint['SU2_CONFIG']
        weight_line = readConfig(adjoint_flow_config, 'OBJECTIVE_WEIGHT', False)

        if weight_line == 'NO':
            weights = [1.0]
        else:
            weights = [float(w) for w in weight_line.replace(',', ' ').split()]

        if self.opt_mode == "AERO":

            if any(w != 1.0 for w in weights):
                sys.exit('OBJECTIVE_WEIGHT in ' + adjoint_flow_config +
                          ' must be 1.0 (or absent/commented out) when OPT_MODE = AERO.'
                          ' Found: ' + weight_line)

            print('Optimisation case: AERO objective / AERO design variables')

        elif self.opt_mode == "STRUCT":

            if all(w == 1.0 for w in weights):
                print('Optimisation case: AERO objective / STRUCT design variables')

            elif all(w == 0.0 for w in weights):
                print('Optimisation case: STRUCT objective / STRUCT design variables')

            else:
                sys.exit('OBJECTIVE_WEIGHT in ' + adjoint_flow_config +
                          ' must be either 1.0 (or absent/commented out, for an AERO objective) or 0.0'
                          ' (for a STRUCT objective) when OPT_MODE = STRUCT. Found: ' + weight_line)


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
        command = 'cp ' + config_deform + ' ' + self.deform_folder + '/'
        run_command(command, 'Pulling deformation config', False)

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
           command = 'cp ' + config_geo + ' ' + self.geo_folder + '/'
           run_command(command, 'Pulling geo config', False)     
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




    def con_struct_dcieq_normalized(self):

       """
       Solve the coupled adjoint problem for strutural optimisation constraint
       """          
      
       # create adjoint constraint subfolders and pull the needed files
       self.structProject.constr_subfolders_adjoint = []

       for i in range(len(self.structProject.config['AUGUSTO_CONFIG_CONSTR'])):

            # create current adjoint constraint subfolder
            current_adj_folder = self.structProject.design_folder_adjoint + '/' + self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i].split('.')[0]
            self.structProject.constr_subfolders_adjoint.append(current_adj_folder)

            MakeDir(current_adj_folder, 'Creating subdirectory for constraint ' + self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i])
 
            # pull files for analysis (includes the adjoint config itself)
            PullingPrimalAdjointFiles(self.testcase_folder, current_adj_folder, self.configFSIAdjoint, self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i], self.pyInterfaceFile)

            # pulling mesh file 
            self.SetMesh(current_adj_folder)         
       
            # pulling restart for pyAugusto and SU2 and flow.vtk
            if self._design[self.design_iter].primal == True:

               PullRestartFiles(self.primal_folder, current_adj_folder)

            else:
              print('Primal not yet available, can t pull solutions for Adjoint....')
              sys.exit()

            # Running adjoint
            self._design[self.design_iter].FSIAdjoint(current_adj_folder, self.structProject.config['AUGUSTO_CONFIG_CONSTR'][i])

       

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
              