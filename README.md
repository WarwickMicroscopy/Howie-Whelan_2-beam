# Howie-Whelan_2-beam
Basic transmission electron microscope image simulations of disloctions (diffraction contrast) based on the 2-beam Howie-Whelan equations.

This is a more-or-less functioning 2-beam diffraction contrast dislocation image simulator.

The code works for cubic crystals only, avoiding the hassle of converting non-cubic cell coordinates to match the simulation volume.  It uses the column approximation, assuming that each pixel can be calculated individually.  Dislocation displacements are for infinite straight dislocations and take no account of surface relaxation. Hydrostatic expansion/contraction is also ignored.  Multiple beam effects are important in many cases (e.g. weak beam imaging), which can't be accommodated in this simulation.  It is pretty basic. Calculation time is proportional to the number of voxels, i.e. crystal thickness and the number of pixels. My laptop manages a voxel in about 0.5 ms.

The dislocation lies horizontally with the electron beam into the image (i.e. the same way as seen on a conventional TEM screen).  Its intersection with the top surface is to the left, the intersection with the bottom surface to the right.
Version v1.4 has two dislocations rather than one (gives the same result as v1.3 if the Burgers vector of the second dislocation is set to zero).  Version v1.5 adds the stacking fault contrast between the dislocations if they are partial dislocations and the ability to change the number of pixels/nm 


*** Inputs are between lines 90 and 145 and should be fairly obvious to complete.

1) Extinction distance Xg can be obtained from tables, or calculated using scattering factors.  The imaginary parts Xgi and X0i determine the absorption behaviour.  A typical extinction distance for imaging dislocations in fcc materials is between 20 and 30 nm.
2) Thickness t.  In nm.
3) Integration step dt, also in nm. The calculation is done in slices of thickness dt. dt>1 speeds up the calculation in proportion (dt=5 is 5 times faster than dt=1) BUT there will be artefacts in the calculation next to the dislocation line.  Useful for a quick setup run to make sure things work as expected.  There is little beneftit from having dt<0.5.
4) pix2nm (v1.5) Number of pixels/nm (the default is 1.0). Values higher than 1 will produce proportionally fewer pixels than specified in pad and picmax, and will complete quickly (quadratic dependence).  Values smaller than 1 will give more detail and complete more slowly. 
5) Padding pad, in pixels.  The image height is 2*pad; there is also padding to the left and right of the intersection points.
6) Maximum image size picmax, assuming 1 pix/nm.  Will over-rule pad.
7) Lattice parameter a0, in nm.
8) Poisson's ratio nu.
9) Burgers vector b in the usual Miller indices (fractional unit cell coordinates).  In version v1.4 onwards there are TWO parallel dislocations with Burgers vector b1 and b2.
10) sep (v1.5) half of the separation of the two dislocations (relative to the centre of the simulation); dislocation 1 is shifted up by sep, 2 is shifted down by sep.
11) Line direction u in Miller indices
12) Foil normal z in Miller indices
13) g-vector g in Miller indices

Images are saved as 32-bit .tif files and a .png using the default path.

New version v2.1 includes an openCL speed up written by Jon Peters which is about 50,000x faster on a Dell Inspiron 7577 laptop.  Requires funcs_richard and funcs_opencl. Input is now between lines 30 and 100.
