#!/bin/bash

export WDIR=$1
export scenario=$2

export PDADIR=/Users/fnrizzi/Desktop/work/ROM/gitrepos/pressio-demoapps/

# fom needs to be run always
python3 main_run_fom.py --wdir ${WDIR} --pdadir ${PDADIR} \
	--problem py_problems.2d_swe -s ${scenario}

# # the following are all executed AFTER fom data is generated
# # note that any of the following scripts runs something
# # only if it is present in the scenario.
# # For example, let's say your problem/scenario only specifies to
# # run standard pod galerkin, then "main_full_domain_sample_mesh.py" is a noop
# # So bottom line, leave them all here that it does not cost anything

# standard galerkin
python3 main_compute_pod_for_full_domain.py     --wdir ${WDIR}
python3 main_compute_proj_errors_for_full_domain.py --wdir ${WDIR}
python3 main_compute_sample_meshes_for_full_domain.py --wdir ${WDIR} --pdadir ${PDADIR}
python3 main_run_pod_standard_galerkin.py --wdir ${WDIR}

# pod od galerkin
python3 main_make_partitions.py                   --wdir ${WDIR}
python3 main_compute_pod_for_all_partitions.py    --wdir ${WDIR}

python3 main_run_pod_od_galerkin.py               --wdir ${WDIR} --pdadir ${PDADIR}
python3 main_run_pod_od_galerkin_gappy_real.py    --wdir ${WDIR}
python3 main_run_pod_od_galerkin_gappy_masked.py  --wdir ${WDIR}

python3 main_compute_proj_errors_for_od.py        --wdir ${WDIR}
python3 main_compute_od_rom_errors.py             --wdir ${WDIR}
