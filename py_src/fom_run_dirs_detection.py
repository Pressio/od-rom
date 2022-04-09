
import numpy as np
import sys, os, re, yaml

from .miscellanea import get_run_id

# -------------------------------------------------------------------
def find_fom_train_dirs_for_target_set_of_indices(workDir, trainIndices):
  trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
               if "fom_train" in d and get_run_id(d) in trainIndices]
  assert(len(trainDirs) == len(trainIndices))
  return trainDirs

# -------------------------------------------------------------------
def find_all_fom_test_dirs(workDir):
  testDirs = [workDir+'/'+d for d in os.listdir(workDir) \
               if "fom_test" in d]
  return testDirs

# -------------------------------------------------------------------
def find_all_fom_train_dirs(workDir):
  testDirs = [workDir+'/'+d for d in os.listdir(workDir) \
               if "fom_train" in d]
  return testDirs
