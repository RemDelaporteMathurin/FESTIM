from mpi4py import MPI

import basix
import dolfinx
import numpy as np
from dolfinx.mesh import locate_entities
from pyvista import examples

import festim as F

from OF_reader import OpenFOAMReader


my_of_reader = OpenFOAMReader(filename=examples.download_cavity(load=False))
vel = my_of_reader.create_dolfinx_function(t=2.5)

writer = dolfinx.io.VTXWriter(
    MPI.COMM_WORLD, "velocity_points_alt_test.bp", vel, engine="BP5"
)
writer.write(t=0)


class TopSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(
            mesh, fdim, lambda x: np.logical_and(np.isclose(x[1], 0.1), x[0] < 0.05)
        )
        return indices


class BottomSurface(F.SurfaceSubdomain):
    def locate_boundary_facet_indices(self, mesh):
        fdim = mesh.topology.dim - 1
        indices = locate_entities(mesh, fdim, lambda x: np.isclose(x[1], 0))
        return indices


class FluiDomain(F.VolumeSubdomain):
    def locate_subdomain_entities(self, mesh):
        indices = locate_entities(
            mesh,
            3,
            lambda x: x[0] <= of_x + 1e-16,
        )
        return indices


class SolidDomain(F.VolumeSubdomain):
    def locate_subdomain_entities(self, mesh):
        indices = locate_entities(
            mesh,
            3,
            lambda x: x[0] > of_x - 1e-16,
        )
        return indices


my_model = F.HTransportProblemDiscontinuous()

of_x, of_y, of_z = 0.1, 0.1, 0.01
solid_x = 0.01

new_mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[0, 0, 0], [of_x + solid_x, of_y, of_z]], n=[11, 10, 3]
)


# interpolate velocity on the new mesh
element = basix.ufl.element("Lagrange", new_mesh.ufl_cell().cellname(), 1, shape=(3,))
V = dolfinx.fem.functionspace(new_mesh, element)
new_vel = dolfinx.fem.Function(V)
F.helpers.nmm_interpolate(new_vel, vel)


# I want to time the values by 50
new_vel.x.array[:] *= 50

# my_model.mesh = F.Mesh(mesh=my_of_reader.dolfinx_mesh)
my_model.mesh = F.Mesh(mesh=new_mesh)

mat_fluid = F.Material(D_0=0.1, E_D=0, K_S_0=1, E_K_S=0)
mat_solid = F.Material(D_0=0.1, E_D=0, K_S_0=3, E_K_S=0)
vol_fluid = FluiDomain(id=1, material=mat_fluid)
vol_solid = SolidDomain(id=2, material=mat_solid)
top = TopSurface(id=2)
bot = BottomSurface(id=3)

my_model.subdomains = [vol_fluid, vol_solid, top, bot]
my_model.interfaces = [F.Interface(4, (vol_fluid, vol_solid), penalty_term=30)]
my_model.surface_to_volume = {top: vol_fluid, bot: vol_fluid}
H = F.Species("H", mobile=True, subdomains=[vol_fluid, vol_solid])
my_model.species = [H]


my_model.temperature = 500

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top, value=1, species=H),
    F.FixedConcentrationBC(subdomain=bot, value=0, species=H),
]

my_model.exports = [
    F.VTXSpeciesExport(
        filename="test_with_coupling_OF_fluid.bp", field=H, subdomain=vol_fluid
    ),
    F.VTXSpeciesExport(
        filename="test_with_coupling_OF_solid.bp", field=H, subdomain=vol_solid
    ),
]

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, transient=True, stepsize=0.0001, final_time=0.03
)

my_model.advection_terms = [
    F.AdvectionTerm(velocity=new_vel, subdomain=vol_fluid, species=[H]),
]


my_model.initialise()
my_model.run()
