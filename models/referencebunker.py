import numpy as np


from OCP.gp import gp_Pnt
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.TopoDS import TopoDS_Solid, TopoDS_Shape



def create_box(
        x_min: float, 
        y_min: float, 
        z_min: float, 
        dx: float, 
        dy: float, 
        dz: float) -> TopoDS_Shape:
    
    # produce a box using a point coordinates and their deltas
    p1 = gp_Pnt(x_min, y_min, z_min)
    return BRepPrimAPI_MakeBox(p1, dx, dy, dz).Shape()

def generate_bunker_solids() -> list[TopoDS_Solid]:
    wall_thickness: float = 120.0
    
    room_dx: float = 1000.0
    room_dy: float = 600.0
    room_dz: float = 400.0

    room_inner_x: float = -room_dx / 2.0
    room_inner_y: float = -room_dy / 2.0
    room_inner_z: float = -room_dz / 2.0

    corridor_dx: float = room_dx
    corridor_dy: float = 150.0
    corridor_dz: float = 200.0
    
    corridor_inner_x: float = -room_dx / 2.0 - wall_thickness
    corridor_inner_y: float = room_dy / 2.0 + wall_thickness
    corridor_inner_z: float = -room_dz / 2.0 

    # create main bunker
    bunker_shell = create_box(
        room_inner_x - wall_thickness,
        room_inner_y - wall_thickness,
        room_inner_z - wall_thickness,
        room_dx + 2 * wall_thickness,
        room_dy + 2 * wall_thickness,
        room_dz + 2 * wall_thickness
    )

    bunker_cavity = create_box(
        room_inner_x,
        room_inner_y,
        room_inner_z,
        room_dx,
        room_dy,
        room_dz
    )

    corridor_shell = create_box(
        corridor_inner_x,
        corridor_inner_y - wall_thickness,
        corridor_inner_z - wall_thickness,
        corridor_dx + 2 * wall_thickness,
        corridor_dy + 2 * wall_thickness,
        corridor_dz + 2 * wall_thickness
    )

    corridor_cavity = create_box(
        corridor_inner_x,
        corridor_inner_y,
        corridor_inner_z,
        corridor_dx + wall_thickness,
        corridor_dy,
        corridor_dz
    )

    door_cavity = create_box(
        room_dx / 2.0 - corridor_dy,
        room_dy / 2.0,
        -room_dz / 2.0,
        corridor_dy,
        wall_thickness,
        corridor_dz
    )

    # create the main room of the bunker
    bunker_main_room = BRepAlgoAPI_Cut(bunker_shell, bunker_cavity).Shape()

    # create the corridor attached to the side of the bunker
    bunker_corridor = BRepAlgoAPI_Cut(corridor_shell, corridor_cavity).Shape()

    # combine the main room and the corridor
    bunker = BRepAlgoAPI_Fuse(bunker_main_room, bunker_corridor).Shape()

    # cut the door between the main room and the corridor
    bunker = BRepAlgoAPI_Cut(bunker, door_cavity).Shape()

    # create objects to tally
    n_sphere = 5
    offset_from_wall = room_dx / (n_sphere + 1)

    xs = np.linspace((-room_dx / 2.0) + offset_from_wall, (room_dx / 2.0) - offset_from_wall, n_sphere)

    spheres = {f'sphere_{i}': BRepPrimAPI_MakeSphere(gp_Pnt(x, room_dy / 2.0 - offset_from_wall, -room_dz / 2.0 + offset_from_wall), 25.0).Shape() for i, x in enumerate(xs, start=1)}

    # return the final bunker solids and their labels
    return {'bunker': bunker, **spheres}