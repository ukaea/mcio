from mcio.surface import XPlane, YPlane, ZPlane, SphereAtOrigin
from mcio.material import Material
from mcio.cell import Cell
from mcio.source import Point, SourceEnergyDistribution
from mcio.tally import VolumeFluenceTally, PulseHeightTally, FluenceMeshTally
from mcio.mcnp import MCNPRunParameters
from mcio.model import Model

# surfaces
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
vol_graveyard = +surf_sphere


# material
mat_water = Material(1.0,                            # set density to 1 g/cc
                     {'H': 2, 'O': 1},               # set atom fraction
                     name='light water',             # give the material a name (optional)
                     density_type='g/cc',            # set the units of the density
                     fraction_type='ao')             # set the fraction type, either ao for atom or wo for weight

# cells
cell_cube = Cell(volume=vol_cube, material=mat_water, name='cube', comment='cube to be tallied')
cell_void = Cell(volume=vol_void, name='void')
cell_graveyard = Cell(volume=vol_graveyard, name='graveyard', graveyard=True)

# source
energy_distribution = SourceEnergyDistribution([1, 2, 3], [0.1, 0.5, 0.4])
source = Point(energy=energy_distribution)

# tallies
tally_fluence_cell_dose = VolumeFluenceTally(cells=[cell_cube], 
                                             comment='cube cell fluence with dose conversion factor (arb. units)', 
                                             dose_conversion_factors=[[5, 10, 20], [0.15, 0.30, 0.45]])

tally_pulse_height = PulseHeightTally(cells=[cell_cube], comment='cube cell pulse height', energies=[0.1, 0.2, 0.3])

mesh_tally_fluence = FluenceMeshTally(x_positions=[-100, 100],
                                      x_ints=40,
                                      y_positions=[-100, 100],
                                      y_ints=40,
                                      z_positions=[-100, 100],
                                      z_ints=40)

tallies = [tally_fluence_cell_dose,
           tally_pulse_height,
           mesh_tally_fluence]

#run parameters
mcnp_rp = MCNPRunParameters(xsdir='/lustre/restricted_software/mcnp/xs/xsdir_mcnp6.2_all',
                            nps=1e6,
                            cut_particles=['n', 'p', 'e'],
                            physics_cards=['n', 'p', 'e'],
                            modes=['n', 'p', 'e'],
                            stop_tallies={tally_fluence_cell_dose: 0.02})

model = Model(cells=[cell_cube, cell_void, cell_graveyard], 
                         source=source, 
                         mcnp_run_parameters=mcnp_rp, 
                         tallies=tallies)

model.write_mcnp('example_1.i')