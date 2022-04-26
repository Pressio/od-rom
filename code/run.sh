#!/bin/bash

set -e

export PDADIR=$1
export WDIR=$2
export prob=$3
export scenario=$4

# each of the following scripts runs something ONLY if the scenario triggers it
# Bottom line, leave them here, no need to comment them out.

# fom needs to be always first
python3 main_run_fom.py \
	--wdir ${WDIR} --pdadir ${PDADIR} \
	--problem py_problems.${prob} -s ${scenario}

#================================
## global galerkin
#================================
# 1. prepare what is needed
python3 main_compute_pod_for_global_domain.py --wdir ${WDIR}
python3 main_compute_sample_meshes_for_global_domain.py --wdir ${WDIR} --pdadir ${PDADIR}
# 2. proj errors
python3 main_compute_proj_errors_for_global_domain.py --wdir ${WDIR}
# 3. run actual rom
python3 main_global_galerkin_with_pod.py --wdir ${WDIR}

#================================
## od galerkin
#================================
python3 main_create_od_partitions.py --wdir ${WDIR}
python3 main_compute_tile_local_pod.py --wdir ${WDIR}
python3 main_compute_sample_meshes_for_od_domain.py --wdir ${WDIR} --pdadir ${PDADIR}

# 2. proj errors
python3 main_compute_proj_errors_for_od_domain.py --wdir ${WDIR}

# 3. run actual rom and process errors
python3 main_od_galerkin_with_pod.py --wdir ${WDIR} --pdadir ${PDADIR}
python3 main_od_gappy_galerkin_with_pod.py --wdir ${WDIR}
python3 main_od_gappy_masked_galerkin_with_pod.py --wdir ${WDIR}
python3 main_od_quad_galerkin_with_pod.py --wdir ${WDIR}
python3 main_compute_errors_for_od_runs.py --wdir ${WDIR}
