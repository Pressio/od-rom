
import numpy as np

from dictionaries import *

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
    'dt' : 0.02,
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

odrom_use_ic_reference_state[1] = False

odrom_algos[1]      = ["PodGalerkinGappy", "PodGalerkinFull"]

odrom_energies[1]     = [99.999999999]
odrom_basis_sets[1]   = {
  0: [0,1]
}

odrom_partitioning_topol[1] = [[1,1]]
odrom_partitioning_style[1] = ['uniform']

odrom_sample_meshes[1] = [["random", 0.2, 0]]
