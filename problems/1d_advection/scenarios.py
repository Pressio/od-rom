import numpy as np

from problems.dictionaries import *

'''
purely reproductive case: scope is to see if
odrom can do reproductive with
better accurary but with less overall modes than regular rom
'''
base_dic[1] = {
  'fom' : {
    'meshSize': [2000],
    'finalTimeTrain': 4.0,
    'finalTimeTest' : 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001,
    'stateSamplingFreqTrain'    : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001,
  },

  'stateSamplingFreqTest' : 2,

  'physicalCoefficients' : {
    'velocity'    : "tbd"
  }
}

train_points[1] = { 0: 1.2 }
test_points[1]  = { 0: 1.2 }

basis_sets[1] = { 0: [0] }
use_ic_reference_state[1] = True

algos[1] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull"]

standardrom_energies[1] = [99.999, 99.9999, 99.99999, 99.999999, 99.9999999, 99.99999999]
odrom_energies[1]       = [99.99, 99.999, 99.9999, 99.99999, 99.999999, 99.9999999]

odrom_partitions[1] = { 'rectangularUniform' : [2,4,6,8,10] }


# ================================================

'''
fix velocity, do prediction in time only.
It is important here to use only a single train
point because if we were to use two or more,
we would increase the predictive capabilities of the basis.
Keeping a single train point, we somewhat test the
intrinsic predictive capabilities of the method
'''
base_dic[2] = {
  'fom' : {
    'meshSize': [2000],
    'finalTimeTrain': 4.0,
    'finalTimeTest' : 4.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001,
    'stateSamplingFreqTrain'    : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 4.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001
  },

  'stateSamplingFreqTest' : 2,

  'physicalCoefficients' : {
    'velocity'    : "tbd"
  }
}

train_points[2] = { 0: 1.2 }
test_points[2]  = { 0: 1.2 }

basis_sets[2]   = { 0: [0] }
use_ic_reference_state[2] = True

algos[2] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull",
            "PodOdGalerkinGappy"]

standardrom_energies[2] = [99.999, 99.9999, 99.99999, 99.999999, \
                           99.9999999, 99.99999999, 99.999999995, \
                           99.999999999, 99.9999999995, 99.9999999999]
odrom_energies[2]       = [99.99, 99.995, 99.999, 99.9995, \
                           99.9999, 99.99995, \
                           99.99999, 99.999995, \
                           99.999999, 99.9999995, \
                           99.9999999, 99.99999995]

odrom_partitions[2] = { 'rectangularUniform' : [2,4,6,8,10,20] }
sample_meshes[2] = [["psampling", 0.25, 0]]


# ================================================

'''
vary velocity, fix simulation time
'''
base_dic[3] = {
  'fom' : {
    'meshSize': [1000],
    'finalTimeTrain': 4.0,
    'finalTimeTest' : 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001,
    'stateSamplingFreqTrain' : 2,
    'velocitySamplingFreq'   : 2
  },

  'odrom' : {
    'finalTime': 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001
  },

  'stateSamplingFreqTest' : 5,

  'physicalCoefficients' : {
    'velocity'    : "tbd"
  }
}

train_points[3] = {
  0: 0.6,
  1: 1.6
}

test_points[3]  = {
  # 0: 0.2,
  # 1: 0.7,
  2: 1.3,
  3: 1.8,
  4: 2.0
}

basis_sets[3]   = {
  0: [0,1]
}

use_ic_reference_state[3] = True

algos[3] = ["PodStandardGalerkinFull", \
            "PodOdGalerkinFull"]

standardrom_energies[3] = [99.99999, 99.9999999, 99.9999999999]
odrom_energies[3]       = [99.999, 99.999999, 99.999999999]

odrom_partitions[3] = { 'rectangularUniform' : [4, 8] }
