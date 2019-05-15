#!/usr/bin/env python

## \file FSI_config.py
#  \brief Python class for handling configuration file for FSI computation.
#  \author David Thomas
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

import os, sys, shutil, copy
import ast
from util import switch

# ----------------------------------------------------------------------
#  MLS Configuration Class
# ----------------------------------------------------------------------

class MLSConfig:
    """
    Class that contains all the parameters coming from the MLS configuration file.
    Read the file and store all the options into a dictionary.
    """

    def __init__(self,FileName):
        self.ConfigFileName = FileName
        self._ConfigContent = {}
        self.readConfig()

    def __str__(self):
	tempString = str()
	for key, value in self._ConfigContent.items():
	   tempString += "{} = {}\n".format(key,value)
	return tempString

    def __getitem__(self,key):
	return self._ConfigContent[key]

    def __setitem__(self, key, value):
	self._ConfigContent[key] = value

    def readConfig(self):
        input_file = open(self.ConfigFileName)
        while 1:
	    line = input_file.readline()
	    if not line:
	        break
            # remove line returns
            line = line.strip('\r\n')
            # make sure it has useful data
            if (not "=" in line) or (line[0] == '%'):
                continue
            # split across equal sign
            line = line.split("=",1)
            this_param = line[0].strip()
            this_value = line[1].strip()

            for case in switch(this_param):
	        #integer values
		if case("MODES")			      : pass
                if case("POLY")		                      : pass
                if case("WEIGHT")		              : pass
                if case("POINTS")		              : 
		    self._ConfigContent[this_param] = int(this_value)
		    break

	        #float values
		if case("MAGNIF_FACTOR")	      : pass                
                if case("RMAX")		              : pass                
                if case("DELTA")                      : pass
                if case("TOLL_SVD")                   : pass
                #if case("DECOMP_COEFF")		      : pass
		if case("FREQ")		              : 
		    self._ConfigContent[this_param] = float(this_value)
		    break

                #float array
                if case("DECOMP_COEFF")                      :
                    this_value1 = ast.literal_eval(this_value)
                    self._ConfigContent[this_param] = this_value1
		    break
                    
	        #string values
		if case("STRUCTURAL_NODES_FILE_NAME")	      : pass
                if case("FORMAT_MODES")	                      : pass
                if case("FORMAT_SRUCT_NODES")	              : pass
                if case("DEBUG")	                      : pass
		if case("STRUCTURAL_MODES_FILE_NAME")         : 
	        #if case("MESH_DEF_METHOD")	      : pass
		    self._ConfigContent[this_param] = this_value
		    break

 	        if case():
		    print(this_param + " is an invalid option !")
		    break
	    #end for
	

