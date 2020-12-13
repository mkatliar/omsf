# Description
OMSF stands for **O**ffline **M**otion **S**imulation **F**ramework.

OMSF is a software to calculate optimal trajectories and design parameters for a given motion 
simulator and a set of maneuvers. Models of human perception and
visual-vestibular integration can be taken into account.

For more information, refer to the [OMSF paper](https://www.sciencedirect.com/science/article/abs/pii/S1369847818308544).
```biblatex
@article{KATLIAR201929,
title = "Offline motion simulation framework: Optimizing motion simulator trajectories and parameters",
journal = "Transportation Research Part F: Traffic Psychology and Behaviour",
volume = "66",
pages = "29 - 46",
year = "2019",
issn = "1369-8478",
doi = "https://doi.org/10.1016/j.trf.2019.07.019",
url = "http://www.sciencedirect.com/science/article/pii/S1369847818308544",
author = "Mikhail Katliar and Mario Olivari and Frank M. Drop and Suzanne Nooij and Moritz Diehl and Heinrich H. Bülthoff"
}
```

# Installation
It is recommended to install OMSF in a virtual environment. To prepare the virtual environment, do
```bash
virtualenv -p python3 .venv
source .venv/bin/activate
```
Then, install OMSF using pip:
```bash
pip install git+ssh://git@github.com/mkatliar/omsf.git#egg=omsf
```

OMSF relies on the `Ipopt` optimizer, which in turn uses linear solvers.
For better performance, it is recommended to install the `HSL` library that contains high-performance linear solvers: https://github.com/casadi/casadi/wiki/Obtaining-HSL