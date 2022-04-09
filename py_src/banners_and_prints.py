
import numpy as np
import sys, os

def color_example():
  # https://ozzmaker.com/add-colour-to-text-in-python/

  print("\033[0;37;40m Normal text\n")
  print("\033[2;37;40m Underlined text\033[0;37;40m \n")
  print("\033[1;37;40m Bright Colour\033[0;37;40m \n")
  print("\033[3;37;40m Negative Colour\033[0;37;40m \n")
  print("\033[5;37;40m Negative Colour\033[0;37;40m\n")
  print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
  print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
  print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
  print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
  print("\033[1;33;40m Yellow         \033[0m 1;33;40m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
  print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
  print("\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
  print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
  print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")

def check_and_print_problem_summary(problem, module):
  try:
    print("{}: dimensionality = {}".format(problem, module.dimensionality))
  except:
    sys.exit("Missing dimensionality in problem's module")

  try:
    print("{}: numDofsPerCell = {}".format(problem, module.numDofsPerCell))
  except:
    sys.exit("Missing numDofsPerCell in problem's module")

  print(module)

def color_resetter():
  return "\033[0;0m"

def print_separator():
  print("\033[1;30;47m-------------------------------------------------")

def print_separator_with_resetter():
  print("\033[1;30;47m-------------------------------------------------" + \
        color_resetter())

def banner_import_problem():
  print_separator()
  print("\033[1;30;47m    Importing problem module                     ")
  print_separator_with_resetter()

def banner_compute_full_pod():
  print_separator()
  print("\033[1;30;47m    Compute FULL domain POD                      ")
  print_separator_with_resetter()

def banner_make_fom_mesh():
  print_separator()
  print("\033[1;30;47m    Make FOM mesh                                ")
  print_separator_with_resetter()

def banner_fom_train():
  print_separator()
  print("\033[1;30;47m    FOM Train runs                               ")
  print_separator_with_resetter()

def banner_fom_test():
  print_separator()
  print("\033[1;30;47m    FOM Test runs                                ")
  print_separator_with_resetter()

def banner_sample_mesh_full_domain():
  print_separator()
  print("\033[1;30;47m    Make sample mesh for FULL domain             ")
  print_separator_with_resetter()

def banner_pod_standard_galerkin():
  print_separator()
  print("\033[1;30;47m    Running pod standard-galerkin on FULL domain ")
  print_separator_with_resetter()

def banner_make_partitions():
  print_separator()
  print("\033[1;30;47m    Partitioning domain                          ")
  print_separator_with_resetter()

def banner_compute_pod_all_partitions():
  print_separator()
  print("\033[1;30;47m    Compute POD for all partitions               ")
  print_separator_with_resetter()

def banner_make_sample_meshes_all_partitions():
  print_separator()
  print("\033[1;30;47m    Make sample mesh for all partitions          ")
  print_separator_with_resetter()

def banner_make_full_meshes_with_partition_based_indexing():
  print_separator()
  print("\033[1;30;47m    Make full meshes based on partition indexing ")
  print_separator_with_resetter()

def banner_run_pod_od_galerkin():
  print_separator()
  print("\033[1;30;47m    Run pod od Galerkin                          ")
  print_separator_with_resetter()
