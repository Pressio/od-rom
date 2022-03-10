import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

# True/False to set initial condition as ref state
odrom_use_ic_reference_state = {}

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_algos           = {}
odrom_poly_order      = {}

odrom_partitioning_topol = {}
odrom_partitioning_style = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "1d_advection",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 2.0,
    'finalTimeTest' : 2.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.001,
    'stateSamplingFreq' : 5,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 2.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK2",
    'dt' : 0.005,
    'stateSamplingFreq' : 5
  },

  'physicalCoefficients' : {
    'velocity'    : "tbd"
  }
}

train_points[1] = {
  0: 0.6,
  1: 1.8
}

test_points[1]  = {
  0: 1.2,
  1: 2.0,
  2: 2.2
}

odrom_use_ic_reference_state[1] = True

odrom_algos[1]        = ["PodGalerkinFull", "PolyGalerkinFull"]
odrom_energies[1]     = [99., 99.99, 99.9999, 99.99999, 99.9999999, 99.9999999999, 99.9999999999999, 100.0]
odrom_basis_sets[1]   = {
  0: [0,1]
}

# -1: compute orders of the poly bases to match pod modes and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[1]   = [-1]

odrom_partitioning_topol[1] = [[1],[5],[10],[20],[50],[100]]
odrom_partitioning_style[1] = ['uniform']
