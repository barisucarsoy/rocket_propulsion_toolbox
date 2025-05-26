from build123d import *
from ocp_vscode import show_all
import webbrowser

a, b = 40, 4
with BuildPart() as ex13:
    Cylinder(radius=50, height=10)
    with Locations(ex13.faces().sort_by(Axis.Z)[-1]):
        with PolarLocations(radius=a, count=8):
            Hole(radius=b)
# export_step(ex13.part, "ex13.step")
show_all()

# start the ocp viewer in standalone mode with the following command:
# python -m ocp_vscode
