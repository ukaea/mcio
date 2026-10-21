User Guide 1: CSG
************

```{warning}
MCIO uses MCNP default units: cm for length, MeV for energy etc.
```

# Writing using CSG

## Import modules

```python
# import the classes for CSG model creation and writing
from mcio.surface import XPlane, YPlane, ZPlane, SphereAtOrigin
from mcio.material import Material
from mcio.cell import Cell
from mcio.source import Point, SourceEnergyDistribution
from mcio.tally import VolumeFluenceTally, PulseHeightTally, FluenceMeshTally
from mcio.mcnp import MCNPRunParameters
from mcio.model import Model
```

## Volumes

The constructive solid geometry (CSG) method uses 2D geometric surfaces such as planes, sphere, cylinder, cones and tori, to bound a 3D space.

First we must define 2D geometric surfaces and then convert these into a half-space using the '+' or '-' operations to denote which side of the surface we would like to bound with. 

Secondly we combine these half spaces using boolean operations, either the intersection "&" operator to find common space between two or more 3D volumes, or the union "|" operator which merges the two volumes. 

The example below will create a cube which we will later fill with water, and also a void space surrounding the cube and graveyard volume outside of sphere 

```python
# create surfaces for a cube
surf_x_min = XPlane(x0=-10)
surf_x_max = XPlane(x0=10)
surf_y_min = YPlane(y0=-10)
surf_y_max = YPlane(y0=10)
surf_z_min = ZPlane(z0=-10)
surf_z_max = ZPlane(z0=10)
surf_sphere = SphereAtOrigin(radius=100.0)

# combine surfaces using half space and boolean operations
vol_cube = +surf_x_min & -surf_x_max \
         & +surf_y_min & -surf_y_max \
         & +surf_z_min & -surf_z_max

# we can take the inverse using the ~ operation before a volume or half-space
vol_void = -surf_sphere & ~vol_cube

# define a graveyard
vol_graveyard = +vol_sphere
```

## Materials

We can materials with compositions be either elements or isotopes, and densities. We can specify weight or atom fractions for these compositions. The elements will be expanded using the NIST natural compositions.

```python
# create a material for water
mat_water = Material(1.0,                            # set density to 1 g/cc
                     {'H': 2, 'O': 1},               # set atom fraction
                     name='light water',             # give the material a name (optional)
                     density_type='g/cc',            # set the units of the density
                     fraction_type='ao')             # set the fraction type, either ao for atom or wo for weight
```
## Cells

```Python
# cells
cell_cube = Cell(volume=vol_cube, material=mat_water, name='cube', comment='cube to be tallied')
cell_void = Cell(volume=vol_void, name='void')
cell_graveyard = Cell(volume=vol_graveyard, name='graveyard', graveyard=True)
```

## Sources

```Python
# source
energy_distribution = SourceEnergyDistribution([1, 2, 3], [0.1, 0.5, 0.4])
source = Point(energy=energy_distribution)
```

## Tallies

The native tallies which are included in mcio, and their mappings to MCNP and OpenMC, are:

+-----------------------+--------------------+-------------------------+
| MCIO Tally            | MCNP Tally Mapping | OpenMC Score Mapping    |
+=======================+====================+=========================+
| SurfaceCurrent        | F1 (Total Current) | 'current' (Net Current) |
+-----------------------+--------------------+-------------------------+
| SurfaceFluenceTally   | F2                 | 'flux'                  |
+-----------------------+--------------------+-------------------------+
| VolumeFluenceTally    | F4                 | 'flux'                  |
+-----------------------+--------------------+-------------------------+
| EnergyDepositionTally | F6                 | 'heating'               |
+-----------------------+--------------------+-------------------------+
| PulseHeightTally      | F8                 | 'pulse-height'          |
+-----------------------+--------------------+-------------------------+

Warning: the tallies are in their default forms in each radiation transport code and each code may not handle the tallies equivalently. Notably, for the 'SurfaceCurrent' tally,  MCNP tallies the total current crossing the surface while OpenMC tallies the net current crossing the surface. Equivalence between the tallies can be obtained but that is left to the user. Energies must be input in MeV.

Dose conversion factors can be added to tallies, see the below example.The interpolation between energies and dose coefficients can be 'log-log' (MCNP default), 'log-linear', 'linear-log' or 'linear-linear' (OpenMC default) for MCNP. For OpenMC additional options are available: 'histogram', 'quadratic' or 'cubic'. Refer to the respective documentation for more information. Energies must be in MeV.

```Python
# tallies
tally_fluence_cell_dose = VolumeFluenceTally(cells=[cell_cube], 
                                             comment='cube cell fluence with dose conversion factor (arb. units)', 
                                             dose_conversion_factors=[[5, 10, 20], [0.15, 0.30, 0.45]])

tally_pulse_height = PulseHeightTally(cells=[cell_cube], comment='cube cell pulse height', energies=[0.1, 0.2, 0.3])

mesh_tally_fluence = FluenceMeshTally(x_min=-100, x_max=100, x_ints=40,
                                      y_min=-100, y_max=100, y_ints=40,
                                      z_min=-100, z_max=100, z_ints=40)

tallies = [tally_fluence_cell_dose,
           tally_pulse_height,
           mesh_tally_fluence]
```

There are some tally features which are specific to MCNP and OpenMC which can be set separately. For MCNP these are cell and surface flagging, tally multipliers, tally divisors and dose interpolation. For OpenMC there is dose intrepolation (more to be adde in future to MCIO). Examples of how to define them are shown below. Note that if a single value is set as the tally divisor (as in the example) then that value will be applied to all cell/surfaces in the tally, otherwise a list can be supplied for the divisors.

```Python
# MCNP tally features
tally_fluence_cell_dose.mcnp.cell_flags = [cell_void]
tally_fluence_cell_dose.mcnp.surface_flags = [surf_sphere]
tally_fluence_cell_dose.mcnp.multiplier = 2
tally_fluence_cell_dose.mcnp.tally_divisor = 1
tally_fluence_cell_dose.mcnp.dose_interpolation = 'log-linear'
tally_fluence_cell_dose.openmc_params.dose_interpolation = 'histogram'

```

## MCNP parameters

Optionally, we can control the MCNP run parameters using an MCNPRunParameters instance, which allows us to set all of the usual mode, physics, cut, stop and nps/prdmp cards.

This is not required, as this will be generated by default.

```Python
mcnp_rp = MCNPRunParameters(xsdir='/lustre/restricted_software/mcnp/xs/xsdir_mcnp6.2_all',
                            nps=1e6,
                            cut_particles=['neutron', 'photon', 'electron'],
                            physics_cards=['neutron', 'photon', 'electron'],
                            modes=['neutron', 'photon', 'electron'],
                            stop_tallies={tally_fluence_cell_dose: 0.02})
```

## Model

```Python
model = Model(cells=[cell_cube, cell_void, cell_graveyard], 
                         source=source, 
                         mcnp_run_paramaters=mcnp_rp, 
                         tallies=tallies)

model.write_mcnp('example_1.i')
```