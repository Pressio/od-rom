#!/bin/bash

set -e

export PDADIR=$1
export WDIR=$2
export prob=$3
export scenario=$4

# note that any of the following scripts runs something
# only if it is present in the scenario.
# For example, let's say your problem/scenario only specifies to
# run standard pod galerkin, then "main_full_domain_sample_mesh.py" is a noop
# So bottom line, leave them all here, it does not cost anything.

# fom needs to be always first
python3 main_run_fom.py \
	--wdir ${WDIR} --pdadir ${PDADIR} --problem py_problems.${prob} -s ${scenario}


################################
## standard galerkin
################################
# 1. prepare what is needed
python3 main_compute_pod_for_global_domain.py --wdir ${WDIR}
python3 main_compute_sample_meshes_for_global_domain.py --wdir ${WDIR} --pdadir ${PDADIR}
# 2. proj errors
python3 main_compute_proj_errors_for_global_domain.py --wdir ${WDIR}
# 3. run actual rom
python3 main_pod_global_galerkin.py --wdir ${WDIR}


################################
## pod od galerkin
################################
python3 main_create_od_partitions.py --wdir ${WDIR}
python3 main_compute_pod_for_od_domain.py --wdir ${WDIR}
python3 main_compute_sample_meshes_for_od_domain.py --wdir ${WDIR} --pdadir ${PDADIR}

# 2. proj errors
python3 main_compute_proj_errors_for_od_domain.py --wdir ${WDIR}

# 3. run actual rom and process errors
python3 main_pod_od_galerkin.py		     --wdir ${WDIR} --pdadir ${PDADIR}
python3 main_pod_od_galerkin_gappy_real.py   --wdir ${WDIR}
python3 main_pod_od_galerkin_quad_real.py    --wdir ${WDIR}
python3 main_pod_od_galerkin_gappy_masked.py --wdir ${WDIR}
python3 main_compute_errors_for_od_runs.py   --wdir ${WDIR}
