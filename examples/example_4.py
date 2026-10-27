"""Example 4 converts a CSG volume into an OCP solid, and then saves the
solid to a STEP file.
"""


import mcio

# surfaces
surf_x_plane_min = mcio.surface.XPlane(x0=-50)
surf_x_plane_max = mcio.surface.XPlane(x0=50)
surf_y_plane_min = mcio.surface.YPlane(y0=-50)
surf_y_plane_max = mcio.surface.YPlane(y0=50)
surf_z_plane_min = mcio.surface.ZPlane(z0=-50)
surf_z_plane_max = mcio.surface.ZPlane(z0=50)

surf_sphere = mcio.surface.Sphere(radius=60)
surf_graveyard = mcio.surface.Sphere(radius=100)

# volumes
vol = +surf_x_plane_min & -surf_x_plane_max & \
      +surf_y_plane_min & -surf_y_plane_max & \
      +surf_z_plane_min & -surf_z_plane_max & \
      -surf_sphere

for y in range(-15, 16, 10):
    for z in range(-15, 16, 10):
        surf_cylinder = mcio.surface.CylinderParallelToXAxis(y0=y, z0=z, radius=1.0)
        vol = vol & +surf_cylinder
        
for x in range(-15, 16, 10):
    for z in range(-15, 16, 10):
        surf_cylinder = mcio.surface.CylinderParallelToYAxis(x0=x, z0=z, radius=1.0)
        vol = vol & +surf_cylinder

for x in range(-15, 16, 10):
    for y in range(-15, 16, 10):
        surf_cylinder = mcio.surface.CylinderParallelToZAxis(x0=x, y0=y, radius=1.0)
        vol = vol & +surf_cylinder

# save volume to step
step_file_writer = mcio.ocpfilereadwrite.STEPFileWriter()
step_file_writer.add_solid(vol.ocp_solid(), 'solid_from_csg', rgb=(255,255,0))
step_file_writer.write('model.stp')

# cells
cell_main_body = mcio.cell.Cell(volume=vol)
cell_void = mcio.cell.Cell(volume=~vol & -surf_graveyard)
cell_graveyard = mcio.cell.Cell(volume=+surf_graveyard)

# model 
model = mcio.model.Model(cells=[cell_main_body, cell_void, cell_graveyard])
model.write_mcnp('my_model.i')
model.write_openmc('my_openmc_model')
