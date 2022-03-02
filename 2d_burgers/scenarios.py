import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_algos           = {}
odrom_poly_order      = {}

odrom_partitioning_topol = {}
odrom_partitioning_style = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "2d_burgers",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 5.,
    'finalTimeTest' : 8.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 8.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.01,
    'stateSamplingFreq' : 4
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
  0: 0.3,
  1: 0.65
}

odrom_algos[1]        = ["PodGalerkinFull", "PolyGalerkinFull"]
odrom_energies[1]     = [99.99, 99.999]#, 99.9999, 99.99999]
odrom_basis_sets[1]   = {
  0: [0,1]
}

# -1: compute orders of the poly bases to match pod modes and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[1]   = [-1]

odrom_partitioning_topol[1] = [[3,3], [8,8], [16,16]]
odrom_partitioning_style[1] = ['uniform']


'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
