# Howie-Whelan_2-beam
Basic transmission electron microscope image simulations of disloctions (diffraction contrast) based on the 2-beam Howie-Whelan equations.

This is a more-or-less functioning 2-beam diffraction contrast dislocation image simulator.

The code works for cubic crystals only, avoiding the hassle of converting non-cubic cell coordinates to match the simulation volume.  It uses the column approximation, assuming that each pixel can be calculated individually.  Dislocation displacements are for infinite straight dislocations and take no account of surface relaxation. Hydrostatic expansion/contraction is also ignored.  Multiple beam effects are important in many cases (e.g. weak beam imaging), which can't be accommodated in this simulation.  It is pretty basic. Calculation time is proportional to the number of voxels, i.e. crystal thickness and the number of pixels. My laptop manages a voxel in about 0.5 ms.

The dislocation lies horizontally with the electron beam into the image (i.e. the same way as seen on a conventional TEM screen).  Its intersection with the top surface is to the left, the intersection with the bottom surface to the right.


*** Inputs are between lines 90 and 135 and should be fairly obvious to complete.

1) Extinction distance Xg can be obtained from tables, or calculated using scattering factors.  The imaginary parts Xgi and X0i determine the absorption behaviour.  A typical extinction distance for imaging dislocations in fcc materials is between 20 and 30 nm.
2) Thickness t.  In nm.
3) Integration step dt, also in nm. The calculation is done in slices of thickness dt. dt>1 speeds up the calculation in proportion (dt=5 is 5 times faster than dt=1) BUT there will be artefacts in the calculation next to the dislocation line.  Useful for a quick setup run to make sure things work as expected.  There is little beneftit from having dt<0.1.
4) Padding pad, in pixels.  The image height is 2*pad; there is also padding to the left and right of the intersection points.
5) Picmax - the maximum image size.  Will over-rule pad.
6) Lattice parameter a0, in nm.
7) Poisson's ratio nu.
8) Burgers vector b in the usual Miller indices (fractional unit cell coordinates).  In version v1.4 there are TWO parallel dislocations with Burgers vector b1 and b2.
8a) In version v1.4 the position of the two dislocations relative to the centre of the simulation volume are given by coordinates q1 and q2.
9) Line direction u in Miller indices
10) Foil normal z in Miller indices
11) g-vector g in Miller indices

Images are saved as 32-bit .tif files and a .png using the default path.


Very significant speed up (10^3 - 10^5) would be possible by using a more efficient integration routine, linear combination of solutions, and parallelisation on a pixel basis using a GPU.  Maybe to be implemented at a future time.
