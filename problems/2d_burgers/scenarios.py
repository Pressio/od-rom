import numpy as np

from problems.dictionaries import *


'''
-------------------------------------------------------
predict in param only
-------------------------------------------------------
'''
base_dic[1] = {

  'fom' : {
    'meshSize': [160, 160],
    'finalTimeTrain': 5.,
    'finalTimeTest' : 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2
  },

  'physicalCoefficients' : {
    'pulsemag'    : 0.5,
    'pulsespread' : "tbd",
    'diffusion'   : 0.0001,
    "pulsecenter" : [-0.15, -0.3]
  }
}

train_points[1] = {
  0: 0.2,
  1: 0.5
}

test_points[1]  = {
  0: 0.2,
  1: 0.3,
  2: 0.15,
  3: 0.55
}

use_ic_reference_state[1] = True

basis_sets[1] = { 0: [0,1] }

algos[1] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[1] = [99.999, 99.9999999, 99.999999999]
odrom_energies[1] = [99.9999, 99.99999]

odrom_partitioning_topol[1] = [[5,5]]
odrom_partitioning_style[1] = ['uniform']

sample_meshes[1] = [["psampling", 0.25, 0],
                    ["psampling", 0.35, 0]]


'''
-------------------------------------------------------
predict in param and time
-------------------------------------------------------
'''
base_dic[2] = {

  'fom' : {
    'meshSize': [160, 160],
    'finalTimeTrain': 5.,
    'finalTimeTest' : 6.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 6.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2
  },

  'physicalCoefficients' : {
    'pulsemag'    : 0.5,
    'pulsespread' : "tbd",
    'diffusion'   : 0.0001,
    "pulsecenter" : [-0.15, -0.3]
  }
}

train_points[2] = {
  0: 0.2,
  1: 0.5
}

test_points[2]  = {
  0: 0.3
}

use_ic_reference_state[2] = True

basis_sets[2] = { 0: [0,1] }

algos[2] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[2] = [99.999, 99.9999, 99.9999999]
odrom_energies[2] = [99.999, 99.9999, 99.99999]

odrom_partitioning_topol[2] = [[5,5]]
odrom_partitioning_style[2] = ['uniform']

sample_meshes[2] = [["psampling", 0.35, 0]]
