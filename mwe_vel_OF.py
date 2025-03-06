from __future__ import annotations
from dolfinx import mesh as dmesh
import numpy as np
from mpi4py import MPI
import pyvista
from pyvista import examples
import festim as F
import ufl
import basix
import dolfinx
from dolfinx.mesh import locate_entities


class OpenFOAMReader:
    def __init__(self, filename):
        self.filename = filename

        self.dolfinx_mesh = None

    def read_with_pyvista(self):
        self.reader = pyvista.POpenFOAMReader(filename)

    def create_dolfinx_mesh(self):
        pass

    def create_dolfinx_function(self, t):
        pass


my_openfoam_reader = OpenFOAMReader(filename="blabla")

filename = examples.download_cavity(load=False)
reader = pyvista.POpenFOAMReader(filename)
reader.set_active_time_value(2.5)
reader.cell_to_point_creation = True  # Need point data for streamlines
mesh = reader.read()
internal_mesh = mesh["internalMesh"]


cell_data = internal_mesh.cell_data["U"]  # Example for cell_data

# Extract points and connectivity
points = internal_mesh.points
cells = internal_mesh.cells_dict  # Dictionary mapping cell type to connectivity

# Assume tetrahedral mesh (modify based on actual mesh type)
hex_cells = cells.get(12)  # 12 corresponds to VTK_HEXAHEDRON
if hex_cells is None:
    raise ValueError("No hexahedron cells found in the mesh.")

## Connectivity of the mesh (topology) - The second dimension indicates the type of cell used
args_conn = np.argsort(hex_cells, axis=1)
rows = np.arange(hex_cells.shape[0])[:, None]
connectivity = hex_cells[rows, args_conn]

# Define mesh element
shape = "hexahedron"
degree = 1
cell = ufl.Cell(shape)

v_cg = basix.ufl.element("Lagrange", cell.cellname(), 1, shape=(3,))
mesh_ufl = ufl.Mesh(v_cg)

# Create Dolfinx Mesh
domain = dmesh.create_mesh(MPI.COMM_WORLD, connectivity, points, mesh_ufl)
domain.topology.index_map(domain.topology.dim).size_global

num_cells = (
    domain.topology.index_map(domain.topology.dim).size_local
    + domain.topology.index_map(domain.topology.dim).num_ghosts
)
v_cg = basix.ufl.element("CG", cell.cellname(), 1, shape=(3,))
V = dolfinx.fem.functionspace(domain, v_cg)
u_alt = dolfinx.fem.Function(V)

num_vertices = (
    domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
)

vertex_map = np.empty(num_vertices, dtype=np.int32)
c_to_v = domain.topology.connectivity(domain.topology.dim, 0)

for cell in range(num_cells):
    vertices = c_to_v.links(cell)
    for i, vertex in enumerate(vertices):
        vertex_map[vertex] = connectivity[domain.topology.original_cell_index][cell][i]

u_alt.x.array[:] = internal_mesh.point_data["U"][vertex_map].flatten()


writer = dolfinx.io.VTXWriter(
    MPI.COMM_WORLD, "velocity_points_test.bp", u_alt, engine="BP5"
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
my_model.mesh = F.Mesh(mesh=domain)

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
    F.AdvectionTerm(velocity=50 * u_alt, subdomain=vol, species=[H]),
    # F.AdvectionTerm(
    #     velocity=my_openfoam_reader.create_dolfinx_function,
    #     subdomain=vol,
    #     species=[H],
    # ),
]


my_model.initialise()
my_model.run()
