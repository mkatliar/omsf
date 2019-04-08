# Description
OMSF stands for **O**ffline **M**otion **S**imulation **F**ramework.

OMSF is a software to calculate optimal trajectories and design parameters for a given motion 
simulator and a set of maneuvers. Models of human perception and
visual-vestibular integration can be taken into account.

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
