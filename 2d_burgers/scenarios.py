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
  },

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
    'odeScheme': "RK2",
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
  0: 0.3
}

odrom_use_ic_reference_state[1] = True

odrom_algos[1]      = ["PodGalerkinFull", "PodGalerkinGappy"]

odrom_energies[1]     = [99.999, 99.9999, 99.99999]
odrom_basis_sets[1]   = { 0: [0,1] }

odrom_partitioning_topol[1] = [[5,5]]
odrom_partitioning_style[1] = ['uniform']

odrom_sample_meshes[1] = [["psampling", 0.35, 0]]
