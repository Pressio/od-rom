import numpy as np

from problems.dictionaries import *

'''
-------------------------------------------------------

reproductive case but with a twist
- use multiple training runs so that we can see if/how
  the dynamics is complex enough that bases get affected
  as compared to using a single training run
- use a larger dt for ROM

-------------------------------------------------------
'''

base_dic[1] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 8.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq'    : 2,
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

train_points[1] = { 0: -3.0, 1: -0.5 }
test_points[1]  = { 0: -3.0, 1: -0.5 }

use_ic_reference_state[1] = True

basis_sets[1] = { 0: [0,1] }

algos[1] = ["PodStandardGalerkinFull", "PodOdGalerkinFull"]

odrom_partitions[1] = {
  'rectangularUniform' : [[4,4]],
  'concentricUniform' : [5]
}

standardrom_energies[1] = [99.9999999]
odrom_energies[1] = [99.99, 99.9999]

sample_meshes[1] = [["psampling", 0.25, 0]]




'''
-------------------------------------------------------

predict in 1d param space (coriolis)

-------------------------------------------------------
'''

base_dic[2] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 8.0,
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

train_points[2] = { 0: -2.0, 1: -1.0 }
test_points[2]  = { 0: -1.5 }

use_ic_reference_state[2] = True

basis_sets[2] = { 0: [0,1] }

algos[2] = ["PodOdGalerkinGappy"]

odrom_partitions[2] = {
  'rectangularUniform' : [[2,2], [4,4]],
  'concentricUniform' : [4]
}

odrom_energies[2] = [99.9999999]

sample_meshes[2] = [["psampling", 0.25, 0]]




'''
-------------------------------------------------------

predict in both param space (Coriolis) and time

-------------------------------------------------------
'''

base_dic[3] = {
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

train_points[3] = {
  0: -2.0,
  1: -0.5
}

test_points[3]  = {
  0: -2.0,
  1: -1.25
}

use_ic_reference_state[3] = True

basis_sets[3] = { 0: [0,1] }

algos[3] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[3] = [99.99, 99.999999]

odrom_partitions[3] = {
  'rectangularUniform' : [[5,5], [8,8]]
}
odrom_energies[3] = [99.999, 99.99999]

sample_meshes[3] = [["psampling", 0.20, 0],
                    ["psampling", 0.25, 0]]
