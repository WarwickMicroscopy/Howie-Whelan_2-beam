# Howie-Whelan_2-beam
Basic transmission electron microscope image simulations of disloctions (diffraction contrast) based on the 2-beam Howie-Whelan equations.

This is a more-or-less functioning 2-beam diffraction contrast dislocation image simulator.
There is a bug when the dislocation has a screw component and it makes a projection at an angle - you will see some odd fringes extending from one side of the dislocation - but otherwise it seems to work.
This bug is not present if the dislocation projection is horizontal or vertical, or has no screw component.

The code works for cubic crystals only, avoiding the hassle of converting non-cubic cell coordinates to match the simulation volume.  It uses the column approximation, assuming that each pixel can be calculated individually.  Dislocation displacements are for infinite straight dislocations and take no account of surface relaxation. Hydrostatic expansion/contraction is also ignored.  Multiple beam effects are important in many cases (e.g. weak beam imaging), which can't be accommodated in this simulation.  It is pretty basic.


*** Inputs are between lines 80 and 125 and should be fairly obvious to complete.

Extinction distances can be obtained from tables, or calculated using scattering factors.  The imaginary parts determine the absorption behaviour.  A typical extinction distance for imaging dislocations in fcc materials is between 20 and 30 nm.

Calculation time is proportional to the number of voxels, i.e. crystal thickness and the number of pixels (roughly, picmax**2).
My laptop manages a voxel in about 0.5 ms. I recommend a small thickness and picmax (t=20nm, picmax = 30nm) for setup runs.

The calculation is done in slices of thickness dt.
dt>1 speeds up the calculation in proportion (dt=5 is 5* faster than dt=1) BUT there will be artefacts in the calculation next to the dislocation line. 

The vectors x and y give the horizontal and vertical edges of the simulation volume as crystal Miller indices.  There is no check that these are actually perpendicular so be careful to make sure they are.

A significant speed up may be possible by increasing the slice thickness on an adaptive basis: if the displacement produced by the dislocation does not change along the beam path this could be done in a single thick slice rather than many thin slices.  Maybe to be implemented at a future time.
