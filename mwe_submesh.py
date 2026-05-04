"""
$$
\Delta u + J = 0 \ \text{,  on} \ \Omega\\
\Delta u_{sub} + v \nabla u_{sub} -J = 0 \text{,  on} \ \Gamma_1\\
J = h_l (u_{sub} - u)
$$


Boundary conditions:

$$
u = 0 \ \text{,  on} \ \Gamma_2 \\
u_{sub} = 1 \ \text{,  on} \  x=0
$$

"""

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
from dolfinx import plot

dx = 1 / 5
L = 100

nx = int(L / dx)
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([L, 1])],
    [nx, 10],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)
vdim = mesh.topology.dim
fdim = mesh.topology.dim - 1

mesh.topology.create_connectivity(fdim, vdim)


# facet meshtags top and bottom
tag_to_marker = {
    1: lambda x: np.isclose(x[1], 0),  # bottom
    2: lambda x: np.isclose(x[1], 1),  # top
    3: lambda x: np.isclose(x[0], 0),  # left
    4: lambda x: np.isclose(x[0], L),  # right
}

facets = np.array([], dtype=np.int64)
tags = np.array([], dtype=np.int32)
for tag, marker in tag_to_marker.items():
    facet_indices = dolfinx.mesh.locate_entities(mesh, fdim, marker)
    facets = np.concatenate((facets, facet_indices))
    tags = np.concatenate((tags, np.full_like(facet_indices, tag, dtype=np.int32)))
facet_tags = dolfinx.mesh.meshtags(mesh, fdim, facets, tags)

cell_tags = dolfinx.mesh.meshtags(
    mesh,
    vdim,
    np.arange(mesh.topology.index_map(vdim).size_local),
    np.ones(mesh.topology.index_map(vdim).size_local, dtype=np.int32),
)

with dolfinx.io.XDMFFile(mesh.comm, "results/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, x=mesh.geometry)

with dolfinx.io.XDMFFile(mesh.comm, "results/cell_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, x=mesh.geometry)

# make submesh of the bottom boundary
submesh, cmap, vmap, nmap = dolfinx.mesh.create_submesh(
    mesh, dim=fdim, entities=facet_tags.find(1)
)
submesh.topology.create_connectivity(0, 1)

# Function spaces and functions
V_bulk = dolfinx.fem.functionspace(mesh, ("CG", 1))
V_sub = dolfinx.fem.functionspace(submesh, ("CG", 1))

W = ufl.MixedFunctionSpace(V_bulk, V_sub)

u = dolfinx.fem.Function(V_bulk)
u.name = "u"
u_sub = dolfinx.fem.Function(V_sub)
u_sub.name = "u_sub"

v, v_sub = ufl.TestFunctions(W)

# Formulation
dx = ufl.dx(domain=mesh, subdomain_data=cell_tags)
ds = ufl.ds(domain=mesh, subdomain_data=facet_tags)

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
F += ufl.inner(ufl.grad(u_sub), ufl.grad(v_sub)) * ds(1)


# advection term NOTE: this seems to be ignored....
# chaning vel_x doesn't change the solution u_sub at the outlet
# we would expect that increasing vel_x would decrease u_sub at the outlet

vel_x = 10
# NOTE: has to be a 2D vector other wise dolfinx complains
# NOTE 2: i tried setting the y component != 0 and I saw a change!
# could it be that the vector is somehow transposed?

# Option 1: Full grad with 2D vector. Works but odd that we need a 2D velocity
vel = dolfinx.fem.Constant(submesh, PETSc.ScalarType([vel_x, vel_x]))
F += ufl.inner(ufl.dot(ufl.grad(u_sub), vel), v_sub) * ds(1)

# Option 2: just du/dx * vel_x. Doesn't work at all
# F += ufl.inner(u_sub.dx(0) * vel_x, v_sub) * ds(1)

# Option 3: just du/dy * vel_x. Works but doesn't make sense since d/dy....
# F += ufl.inner(u_sub.dx(1) * vel_x, v_sub) * ds(1)

# coupling term
h_l = dolfinx.fem.Constant(mesh, 0.4)
flux = h_l * (u - u_sub)

F += flux * v * ds(1)
F += -flux * v_sub * ds(1)

forms = ufl.extract_blocks(F)

# Dirichlet BC left
bc_top_dofs = dolfinx.fem.locate_dofs_topological(
    V_bulk,
    mesh.topology.dim - 1,
    dolfinx.mesh.locate_entities(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[1], 1)
    ),
)
bc_top = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0),
    bc_top_dofs,
    V_bulk,
)

bc_left_dofs = dolfinx.fem.locate_dofs_topological(
    V_sub, 0, dolfinx.mesh.locate_entities(submesh, 0, lambda x: np.isclose(x[0], 0))
)
bc_left = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(1.0),
    bc_left_dofs,
    V_sub,
)
# Nonlinear problem

problem = dolfinx.fem.petsc.NonlinearProblem(
    forms,
    [u, u_sub],
    bcs=[
        bc_top,
        bc_left,
    ],
    petsc_options_prefix="codim1_prob",
    entity_maps=[cmap],
)

problem.solve()
import basix

element = basix.ufl.element("DG", submesh.topology.cell_name(), 0, shape=(1,))
V_grad_sub = dolfinx.fem.functionspace(submesh, element)
u_sub_dx0 = dolfinx.fem.Function(V_grad_sub)
u_sub_dx1 = dolfinx.fem.Function(V_grad_sub)
expr0 = dolfinx.fem.Expression(u_sub.dx(0), V_grad_sub.element.interpolation_points)
expr1 = dolfinx.fem.Expression(u_sub.dx(1), V_grad_sub.element.interpolation_points)
u_sub_dx0.interpolate(expr0)
u_sub_dx1.interpolate(expr1)

print(u_sub_dx0.x.array)
print(u_sub_dx1.x.array)


breakpoint()

# Post processing

with dolfinx.io.VTXWriter(mesh.comm, "results/u.bp", [u]) as writer:
    writer.write(0.0)

with dolfinx.io.VTXWriter(submesh.comm, "results/u_sub.bp", [u_sub]) as writer:
    writer.write(0.0)

topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = u.x.array
grid.set_active_scalars("c")

plotter = pyvista.Plotter()

plotter.add_mesh(grid)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("u.png")

topology, cell_types, geometry = plot.vtk_mesh(u_sub.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["c"] = u_sub.x.array
grid.set_active_scalars("c")

# Make two points to construct the line between
a = [0, 0, 0]
b = [L, 0, 0]
sample = grid.sample_over_line(a, b, resolution=100)

plt.plot(sample["Distance"], sample["c"])
plt.ylim(0, 1)
plt.xlabel("x")
plt.ylabel("u_sub")
plt.show()
