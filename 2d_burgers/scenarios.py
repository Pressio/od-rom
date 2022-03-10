import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

# True/False to set initial condition as ref state
odrom_use_ic_reference_state = {}

odrom_algos           = {}
odrom_partitioning_topol = {}
odrom_partitioning_style = {}
odrom_energies        = {}
odrom_basis_sets      = {}
odrom_sample_meshes   = {}
odrom_poly_order      = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "2d_burgers",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 5.,
    'finalTimeTest' : 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK2",
    'dt' : 0.05,
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
  0: 0.3
}

odrom_use_ic_reference_state[1] = False

odrom_algos[1]        = ["PodGalerkinFull"]
odrom_energies[1]     = [99.999]#, 99.9999, 99.99999]
odrom_basis_sets[1]   = {
  0: [0,1]
}

# -1: compute orders of the poly bases to match pod modes and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[1]   = [-1]

odrom_partitioning_topol[1] = [[8,8]]
odrom_partitioning_style[1] = ['uniform']

odrom_sample_meshes[1] = [["random", 0.25], \
                          ["psampling", 0.25]]
