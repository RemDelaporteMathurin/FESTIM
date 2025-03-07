from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx import mesh as dmesh
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


my_model = F.HydrogenTransportProblem()

new_mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[0, 0, 0], [0.1, 0.1, 0.01]], n=[10, 10, 3]
)


# interpolate velocity on the new mesh
element = basix.ufl.element("Lagrange", new_mesh.ufl_cell().cellname(), 1, shape=(3,))
V = dolfinx.fem.functionspace(new_mesh, element)
new_vel = dolfinx.fem.Function(V)
F.helpers.nmm_interpolate(new_vel, vel)


# my_model.mesh = F.Mesh(mesh=my_of_reader.dolfinx_mesh)
my_model.mesh = F.Mesh(mesh=new_mesh)

my_mat = F.Material(name="mat", D_0=0.1, E_D=0)
vol = F.VolumeSubdomain(id=1, material=my_mat)
top = TopSurface(id=2)
bot = BottomSurface(id=3)

my_model.subdomains = [vol, top, bot]

H = F.Species("H", mobile=True, subdomains=[vol])
my_model.species = [H]


my_model.temperature = 500

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top, value=1, species=H),
    F.FixedConcentrationBC(subdomain=bot, value=0, species=H),
]

my_model.exports = [
    F.VTXSpeciesExport(filename="test_with_coupling_OF.bp", field=H),
]

my_model.settings = F.Settings(
    atol=1e-10, rtol=1e-10, transient=True, stepsize=0.0001, final_time=0.03
)

my_model.advection_terms = [
    F.AdvectionTerm(velocity=50 * new_vel, subdomain=vol, species=[H]),
]


my_model.initialise()
my_model.run()
