
# od-rom

1. install pressio-demoapps

use commit: `8eeb77770b882c3caec7a78b134c929a41fb3343`
see this: https://pressio.github.io/pressio-demoapps/installation.html

2. set up a scenario inside your problem of choice 

  - SWE:
    - ./py_problems/2d_swe/scenarios.py
    - make sure you also edit the init file: ./py_problems/2d_swe/__init__.py

  - Burgers2d:
  	- tbd
  	- tbd

3. run the bach script:

```bash
export WORKDIR=/home/your_test
export PDADIR=<fullpath-to-your-cloned-pressiodemoapps>
bash run.sh ${PDADIR} ${WORKDIR} <scenario-number>
```
