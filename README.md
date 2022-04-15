
# od-rom

1. install pressio-demoapps

  - see this: https://pressio.github.io/pressio-demoapps/installation.html
  - use commit: `8eeb77770b882c3caec7a78b134c929a41fb3343`

2. set up a scenario inside your problem of choice

  - SWE:
    - subdirectory: `py_problems.2d_swe`
    - `./code/py_problems/2d_swe/scenarios.py`
    - make sure you also edit the init file: `./code/py_problems/2d_swe/__init__.py`

  - Gray-Scott:
    - subdirectory: `py_problems.2d_gs`
    - `./code/py_problems/2d_gs/scenarios.py`
    - make sure you also edit the init file: `./code/py_problems/2d_gs/__init__.py`

3. run the bach script:

```bash
export WORKDIR=/home/your_test
export PDADIR=<fullpath-to-your-cloned-pressiodemoapps>
cd code
bash run.sh ${PDADIR} ${WORKDIR} 2d_swe <scenario-number>
```
if you want a different problem, change `2d_swe` accordingly
