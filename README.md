# VBC Line Profile Model

X-ray emission line profiles for OB stars based on the Variable Boundary Condition theory from Gunderson et al. (2022, https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.2663G/abstract).

Provided code is written for the Python wrap around of XSPEC (PyXspec). The model is currently written to be included as a local model so must be loaded with each instance of PyXspec that is started. As written, the VBC model functions can be loaded automatically into your PyXspec session by importing "VBC_Line_Model.py" as a library.

Requirement minimums:

Python 3.10.9

numpy 1.23.5

scipy 1.10.0

XSPEC 12.12

PyXspec 2.1.2

