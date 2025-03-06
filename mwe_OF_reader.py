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


class OpenFOAMReader:
    def __init__(self, filename, OF_mesh_type_value: int = 12):
        """

        Args:
            filename: the filename
            OF_mesh_type_value: cell type id (12 corresponds to HEXAHEDRON)
        """
        self.filename = filename
        self.OF_mesh_type_value = OF_mesh_type_value

        self.OF_mesh = None
        self.dolfinx_mesh = None

        self.reader = pyvista.POpenFOAMReader(self.filename)

    def _read_with_pyvista(self, t: float):
        self.reader.set_active_time_value(t)
        OF_multiblock = self.reader.read()
        self.OF_mesh = OF_multiblock["internalMesh"]

        # Dictionary mapping cell type to connectivity
        OF_cells_dict = self.OF_mesh.cells_dict

        self.OF_cells = OF_cells_dict.get(self.OF_mesh_type_value)
        if self.OF_cells is None:
            raise ValueError(
                f"No {self.OF_mesh_type_value} cells found in the mesh. Found {OF_cells_dict.keys()}"
            )

    def _create_dolfinx_mesh(self):
        ## Connectivity of the mesh (topology) - The second dimension indicates the type of cell used

        args_conn = np.argsort(self.OF_cells, axis=1)
        rows = np.arange(self.OF_cells.shape[0])[:, None]
        self.connectivity = self.OF_cells[rows, args_conn]

        # Define mesh element
        if self.OF_mesh_type_value == 12:
            shape = "hexahedron"
        degree = 1
        cell = ufl.Cell(shape)
        self.mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        mesh_ufl = ufl.Mesh(self.mesh_element)

        # Create Dolfinx Mesh
        self.dolfinx_mesh = dmesh.create_mesh(
            MPI.COMM_WORLD, self.connectivity, self.OF_mesh.points, mesh_ufl
        )
        self.dolfinx_mesh.topology.index_map(self.dolfinx_mesh.topology.dim).size_global

    def create_dolfinx_function(self, t=None) -> dolfinx.fem.Function:
        self._read_with_pyvista(t=t)
        self._create_dolfinx_mesh()
        self.function_space = dolfinx.fem.functionspace(
            self.dolfinx_mesh, self.mesh_element
        )
        u = dolfinx.fem.Function(self.function_space)

        num_vertices = (
            self.dolfinx_mesh.topology.index_map(0).size_local
            + self.dolfinx_mesh.topology.index_map(0).num_ghosts
        )
        vertex_map = np.empty(num_vertices, dtype=np.int32)
        c_to_v = self.dolfinx_mesh.topology.connectivity(
            self.dolfinx_mesh.topology.dim, 0
        )

        num_cells = (
            self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).size_local
            + self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).num_ghosts
        )
        for cell in range(num_cells):
            vertices = c_to_v.links(cell)
            for i, vertex in enumerate(vertices):
                vertex_map[vertex] = self.connectivity[
                    self.dolfinx_mesh.topology.original_cell_index
                ][cell][i]

        u.x.array[:] = self.OF_mesh.point_data["U"][vertex_map].flatten()

        return u


def find_closest_value(values: list[float], target: float) -> float:
    """
    Finds the closest value in a NumPy array of floats to a given target float.

    Parameters:
        values (np.ndarray): Array of float values.
        target (float): The target float value.

    Returns:
        float: The closest value from the array.
    """
    values = np.asarray(values)  # Ensure input is a NumPy array
    return values[np.abs(values - target).argmin()]


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
