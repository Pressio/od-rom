import numpy as np

from problems.dictionaries import *

base_dic[1] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 1000.,
    'finalTimeTest' : 1000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreq' : 5,
    'velocitySamplingFreq' : 5
  },

  'odrom' : {
    'finalTime': 1000.0,
    'odeScheme': "RK4",
    'dt' : 0.4,
    'stateSamplingFreq' : 5
  },

  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : "tbd",
    'feedRate'   : 0.03
  }
}

train_points[1] = { 0: 0.060, 1: 0.065, 2: 0.070 }

test_points[1]  = {
  0: 0.0625,
  1: 0.068,
  2: 0.072
}

basis_sets[1] = { 0: [0,1,2] }

use_ic_reference_state[1] = True

algos[1] = ["PodOdGalerkinFull", "PodOdGalerkinGappy", "PodStandardGalerkinFull"]

standardrom_energies[1] = [99.99999, 99.99999999]
odrom_energies[1]       = [99.9999, 99.9999999, 99.999999999]

odrom_partitions[1] = {
  #'concentricUniform'  : [3, 5, 8],
  'rectangularUniform' : [[5,5], [10,10], [20,20]],
}

sample_meshes[1] = [["psampling", 0.2, 0],
                    ["psampling", 0.2, 1]]

#----------------------------------------------
#----------------------------------------------


base_dic[2] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 1000.,
    'finalTimeTest' : 1000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 4
  },

  'odrom' : {
    'finalTime': 1000.0,
    'odeScheme': "RK4",
    'dt' : 0.4,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : 0.062,
    'feedRate'   : "tbd"
  }
}

train_points[2] = {
  0: 0.03,
  1: 0.05,
  2: 0.07,
}

test_points[2]  = {
  0: 0.04,
  1: 0.06,
  2: 0.075,
  3: 0.025
}

basis_sets[2]   = { 0: [0,1,2] }

use_ic_reference_state[2] = True

algos[2] = ["PodOdGalerkinFull", "PodOdGalerkinGappy"]

standardrom_energies[2] = [99.9999999, 99.999999999]
odrom_energies[2] = [99.9999999]

odrom_partitions[2] = {
  'concentricUniform' : [10]
}

sample_meshes[2] = [["psampling", 0.25, 1]]
