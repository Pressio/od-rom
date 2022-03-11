import numpy as np

from problems.dictionaries import *

'''
-------------------------------------------------------

predict in time only
fix the param value but simulate for longer than training
This is not fully reproductive but close.

-------------------------------------------------------
'''

base_dic[1] = {

  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 6.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.010,
    'stateSamplingFreq' : 2
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : 0.125
  }
}

train_points[1] = { 0: -2.0 }
test_points[1]  = { 0: -2.0 }

use_ic_reference_state[1] = True

basis_sets[1] = { 0: [0] }

algos[1] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[1] = [99.99, 99.999999, 99.9999999999]

odrom_partitioning_style[1] = ['uniform']
odrom_partitioning_topol[1] = [[4,4], [5,5], [8,8]]
odrom_energies[1] = [99.999, 99.99999]

sample_meshes[1] = [["psampling", 0.20, 0],
                    ["psampling", 0.25, 0]]



'''
-------------------------------------------------------

predict in both param space and time

-------------------------------------------------------
'''

base_dic[2] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 6.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 4
  },

  'odrom' : {
    'finalTime': 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.010,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : 0.125
  }
}

train_points[2] = {
  0: -2.0,
  1: -0.5
}

test_points[2]  = {
  0: -2.0,
  1: -1.25
}

use_ic_reference_state[2] = True

basis_sets[2] = { 0: [0,1] }

algos[2] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[2] = [99.99, 99.999999]

odrom_partitioning_style[2] = ['uniform']
odrom_partitioning_topol[2] = [[5,5], [8,8]]
odrom_energies[2] = [99.999, 99.99999]

sample_meshes[2] = [["psampling", 0.20, 0],
                    ["psampling", 0.25, 0]]
