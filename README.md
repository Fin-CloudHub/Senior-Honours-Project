#READ ME

The scripts listed in this repository were used to analyse data from the Quijote simulatino on massive neutrinos to find neutrino wakes.

Neutrino wakes are a non-linear scale phenomenon caused by the gravitational focussing of neutrinos downstream of a moving dark matter halo. This is opposed to the linear scale phenomenon where a dipole in the density and velocity of neutrinos and cold dark matter (CDM) around a moving dark matter halo. This dipole has a high density at the front and a low density at the back and is easier to observe.

The scripts used in the final report are the DensityFields.py and Stacking.py, Stacking.py is used three times, one for each neutrino mass. Be aware that Stacking.py is very slow as it rolls and rotates the whole grid for every halo in all three dimensions one at a time, but produces good results. Cut_out_method.py was an attempt to fix this, by implementing NumPy broadcasting it takes smaller 3D grids around each halo then rotates and slices these all at once. The draw back for this method is the memory space. In order for this method to produce better results it easily requires thousands of gigabytes of memory, attempts were again made to fix this by converting all numpy.float64's to numpy.float32's and to rotate the cut outs in batches. This did not suffice however. The remaining scripts were built during my learning and are unfinished or do not function perfectly.

The Pylians3 python package was used to process the data and is required to run these scripts. For more information on Pylians please refer to: https://github.com/franciscovillaescusa/Pylians3

For more information on the Quijote simulation for massive neutrinos please see: https://quijote-simulations.readthedocs.io/en/latest/Mnu.html

For more information on neutrino wakes you can refer to the following papers:
https://arxiv.org/abs/1311.3422v3
https://arxiv.org/abs/1412.1660v2
https://arxiv.org/abs/1611.04589
https://arxiv.org/abs/2307.00049
https://arxiv.org/abs/1503.07480
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.083518
https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.2093C/abstract
