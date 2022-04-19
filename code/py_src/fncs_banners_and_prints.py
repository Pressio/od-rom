
import numpy as np
import sys, os, logging

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


# -------------------------------------------------------------------
def check_and_print_problem_summary(problem, module):
  logger = logging.getLogger(__name__)

  moduleName = str(module).split(" ")[1]
  logger.info("module name = {}".format(moduleName))
  modulePath = str(module).split(" ")[3]
  logger.info("module path = {}".format(modulePath))

  dimensionality = None
  numDofsPerCell = None
  try:
    dimensionality = module.dimensionality
  except:
    logger.error("Missing dimensionality in problem's module")
    sys.exit(1)

  if dimensionality not in [1,2]:
    logger.error("Invalid dimensionality = {}".format(module.dimensionality))
    sys.exit(1)

  try:
    numDofsPerCell = module.numDofsPerCell
  except:
    logger.error("Missing numDofsPerCell in problem's module")
    sys.exit(1)

  logger.info("dimensionality = {}".format(dimensionality))
  logger.info("numDofsPerCell = {}".format(numDofsPerCell))

# -------------------------------------------------------------------
def color_resetter():
  return "\033[0;0m"

def print_separator():
  print("\033[1;30;47m" + "-"*45 + color_resetter())

# -------------------------------------------------------------------
def banner_driving_script_info(stringToPrint):
  sl = len(stringToPrint)
  tsz = 75 #int(os.get_terminal_size().columns/2)
  estr = " "*int(tsz)
  print("")
  print("\033[1;35;42m" + estr + color_resetter())
  print("\033[1;37;42m " + stringToPrint.upper() + " "*(tsz-sl-1) + color_resetter())
  print("\033[1;35;42m" + estr + color_resetter())

# -------------------------------------------------------------------
def banner_import_problem():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mImporting problem module" + 21*" " + color_resetter())
  # print("\033[1;30;47mImporting problem module                     " \
  #       + color_resetter())

# -------------------------------------------------------------------
# banners for fom related scripts
# -------------------------------------------------------------------
def banner_make_fom_mesh():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mMake FOM mesh"+32*" "+color_resetter())

def banner_fom_train():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mFOM Train runs"+31*" "+color_resetter())

def banner_fom_test():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mFOM Test runs"+32*" "+color_resetter())

# -------------------------------------------------------------------
# global galerkin
# -------------------------------------------------------------------
def banner_compute_full_pod():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mCompute FULL domain POD"+22*" "+color_resetter())

def banner_sample_mesh_full_domain():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mMake sample mesh for full domain"+13*" " +color_resetter())

def banner_pod_standard_galerkin():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod standard-galerkin on full domain     " \
        + color_resetter())

def banner_compute_full_domain_projection_error():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mComputing full domain projection errors      " \
        + color_resetter())

# -------------------------------------------------------------------
# od banners
# -------------------------------------------------------------------
def banner_make_partitions():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mPartitioning domain                          " \
        + color_resetter())

def banner_compute_pod_all_partitions():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mCompute POD for all partitions               " \
        + color_resetter())

def banner_make_sample_meshes_all_partitions():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mMake sample mesh for all partitions          " \
        + color_resetter())

def banner_make_full_meshes_with_partition_based_indexing():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mMake full meshes based on partition indexing " \
        + color_resetter())

def banner_run_pod_od_galerkin():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod od Galerkin                          " \
        + color_resetter())

def banner_compute_od_projection_error():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mComputing pod od projection errors           " \
        + color_resetter())

def banner_run_pod_od_galerkin_gappy_real():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod od Galerkin gappy real               " \
        + color_resetter())

def banner_run_pod_od_galerkin_quad_real():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod od Galerkin quad real                " \
        + color_resetter())

def banner_run_pod_od_galerkin_gappy_masked():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod od Galerkin gappy masked             " \
        + color_resetter())

def banner_run_pod_od_galerkin_quad_real():
  logger = logging.getLogger(__name__)
  logger.info("\033[1;30;47mRun pod od Galerkin quad real                " \
        + color_resetter())
