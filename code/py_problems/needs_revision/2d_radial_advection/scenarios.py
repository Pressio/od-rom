
import numpy as np

from problems.dictionaries import *

base_dic[1] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 3.,
    'finalTimeTest' : 4.5,
    'inviscidFluxReconstruction' : "FirstOrder",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 1,
    'velocitySamplingFreq' : 1
  },

  'odrom' : {
    'finalTime': 4.5,
    'inviscidFluxReconstruction' : "FirstOrder",
    'odeScheme': "RK4",
    'dt' : 0.01,
    'stateSamplingFreq' : 1
  },

  'physicalCoefficients' : {
    'omega' : "tbd"
  }
}

train_points[1] = {
  0: 0.6,
  1: 1.4
}

test_points[1]  = {
  0: 0.8,
  1: 1.6
}

use_ic_reference_state[1] = True

algos[1] = ["PodOdGalerkinGappy"]

odrom_energies[1]     = [99.9999999, 99.999999999, 99.999999999]
basis_sets[1]   = {
  0: [0,1]
}

odrom_partitions[1] = {
  'concentricUniform' : [3,5]
}

sample_meshes[1] = [["psampling", 0.5, 0]]
