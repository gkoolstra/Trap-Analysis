# Trap-Analysis

This set of modules facilitates in loading exported data from Maxwell and Q3D, contains functions to do post processing on the data (e.g. cropping, filtering etc.) and returns useful experimental parameters such as potential curvatures, electron motion frequencies or calculates the cavity frequency shift for a given set of electron positions.

As for now, the module contains 2 files: 

1. `trap_analysis.py` Any functions in this file should be applicable to the small trap region only. For example, the equations of motion for electrons in the trap has the full 2D version implemented, containing derivatives w.r.t. $y$ and $x$.
2. `resonator_analysis.py` Functions in here are applicable for 1D electron motion calculations. In the equations of motion, only $\partial U_{RF}/\partial x$ and $\partial^2 V_{DC}/dx^2$ are implemented. 

An excellent reference for the math used here can be found in the supplement of [this](http://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.011031) paper. 
