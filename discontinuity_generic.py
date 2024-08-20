from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix
from festim.helpers_discontinuity import NewtonSolver, transfer_meshtags_to_submesh
import festim as F


# ---------------- Generate a mesh ----------------
def generate_mesh():
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    def half(x):
        return x[1] <= 0.5 + 1e-14

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 20, 20, dolfinx.mesh.CellType.triangle
    )

    # Split domain in half and set an interface tag of 5
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[top_facets] = 1
    values[bottom_facets] = 2

    bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 4, dtype=np.int32)
    cells[bottom_cells] = 3
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )
    all_b_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(3), tdim, fdim
    )
    all_t_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    interface = np.intersect1d(all_b_facets, all_t_facets)
    values[interface] = 5

    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
    return mesh, mt, ct


mesh, mt, ct = generate_mesh()

material_bottom = F.Material(D_0=2.0, E_D=0.1)
material_top = F.Material(D_0=2.0, E_D=0.1)

material_bottom.K_S_0 = 2.0
material_bottom.E_K_S = 0 * 0.1
material_top.K_S_0 = 4.0
material_top.E_K_S = 0 * 0.12

top_domain = F.VolumeSubdomain(4, material=material_top)
bottom_domain = F.VolumeSubdomain(3, material=material_bottom)
list_of_subdomains = [bottom_domain, top_domain]
list_of_interfaces = {5: [bottom_domain, top_domain]}

top_surface = F.SurfaceSubdomain(id=1)
bottom_surface = F.SurfaceSubdomain(id=2)

H = F.Species("H", mobile=True)
trapped_H = F.Species("H_trapped", mobile=False)
empty_trap = F.ImplicitSpecies(n=0.5, others=[trapped_H])

for species in [H, trapped_H]:
    species.subdomains = [bottom_domain, top_domain]
    species.subdomain_to_solution = {}
    species.subdomain_to_prev_solution = {}
    species.subdomain_to_test_function = {}

list_of_species = [H, trapped_H]

list_of_reactions = [
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=2,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=top_domain,
    ),
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=2,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=bottom_domain,
    ),
]

list_of_bcs = [
    F.DirichletBC(top_surface, value=0.05, species=H),
    F.DirichletBC(bottom_surface, value=0.2, species=H),
]

surface_to_volume = {top_surface: top_domain, bottom_surface: bottom_domain}

V = dolfinx.fem.functionspace(mesh, ("CG", 1))
T = dolfinx.fem.Function(V)
T.interpolate(lambda x: 300 + 10 * x[1] + 100 * x[0])


def D_fun(T, D_0, E_D):
    k_B = 8.6173303e-5
    return D_0 * ufl.exp(-E_D / k_B / T)


def K_S_fun(T, K_S_0, E_K_S):
    k_B = 8.6173303e-5
    return K_S_0 * ufl.exp(-E_K_S / k_B / T)


gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1

num_facets_local = (
    mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
)

for subdomain in list_of_subdomains:
    subdomain.submesh, subdomain.submesh_to_mesh, subdomain.v_map = (
        dolfinx.mesh.create_submesh(mesh, tdim, ct.find(subdomain.id))[0:3]
    )

    subdomain.parent_to_submesh = np.full(num_facets_local, -1, dtype=np.int32)
    subdomain.parent_to_submesh[subdomain.submesh_to_mesh] = np.arange(
        len(subdomain.submesh_to_mesh), dtype=np.int32
    )

    # We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
    # We use the entity on the same side to fix this (as all restrictions are one-sided)

    # Transfer meshtags to submesh
    subdomain.ft, subdomain.facet_to_parent = transfer_meshtags_to_submesh(
        mesh, mt, subdomain.submesh, subdomain.v_map, subdomain.submesh_to_mesh
    )


# Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
# TODO ask Jorgen what this is for
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)
for interface in list_of_interfaces:
    for facet in mt.find(interface):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        for domain in list_of_interfaces[interface]:
            map = domain.parent_to_submesh[cells]
            domain.parent_to_submesh[cells] = max(map)

# ._cpp_object needed on dolfinx 0.8.0
entity_maps = {
    subdomain.submesh: subdomain.parent_to_submesh for subdomain in list_of_subdomains
}


def define_function_spaces(subdomain: F.VolumeSubdomain):
    # get number of species defined in the subdomain
    all_species = [
        species for species in list_of_species if subdomain in species.subdomains
    ]

    # instead of using the set function we use a list to keep the order
    unique_species = []
    for species in all_species:
        if species not in unique_species:
            unique_species.append(species)
    nb_species = len(unique_species)

    degree = 1
    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        subdomain.submesh.basix_cell(),
        degree,
        basix.LagrangeVariant.equispaced,
    )
    element = basix.ufl.mixed_element([element_CG] * nb_species)
    V = dolfinx.fem.functionspace(subdomain.submesh, element)
    u = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)

    us = list(ufl.split(u))
    u_ns = list(ufl.split(u_n))
    vs = list(ufl.TestFunctions(V))
    for i, species in enumerate(unique_species):
        species.subdomain_to_solution[subdomain] = us[i]
        species.subdomain_to_prev_solution[subdomain] = u_ns[i]
        species.subdomain_to_test_function[subdomain] = vs[i]
    subdomain.u = u


def define_formulation(subdomain: F.VolumeSubdomain):
    form = 0
    # add diffusion and time derivative for each species
    for spe in list_of_species:
        u = spe.subdomain_to_solution[subdomain]
        u_n = spe.subdomain_to_prev_solution[subdomain]
        v = spe.subdomain_to_test_function[subdomain]
        dx = subdomain.dx

        D = subdomain.material.get_diffusion_coefficient(mesh, T, spe)
        if spe.mobile:
            form += ufl.inner(D * ufl.grad(u), ufl.grad(v)) * dx

    for reaction in list_of_reactions:
        if reaction.volume != subdomain:
            continue
        for species in reaction.reactant + reaction.product:
            if isinstance(species, F.Species):
                # TODO remove
                # temporarily overide the solution and test function to the one of the subdomain
                species.solution = species.subdomain_to_solution[subdomain]
                species.test_function = species.subdomain_to_test_function[subdomain]

        for reactant in reaction.reactant:
            if isinstance(reactant, F.Species):
                form += (
                    reaction.reaction_term(T)
                    * reactant.subdomain_to_test_function[subdomain]
                    * dx
                )

        # product
        if isinstance(reaction.product, list):
            products = reaction.product
        else:
            products = [reaction.product]
        for product in products:
            form += (
                -reaction.reaction_term(T)
                * product.subdomain_to_test_function[subdomain]
                * dx
            )

    subdomain.F = form


for subdomain in list_of_subdomains:
    define_function_spaces(subdomain)
    ct_r = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        subdomain.submesh_to_mesh,
        np.full_like(subdomain.submesh_to_mesh, 1, dtype=np.int32),
    )
    subdomain.dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct_r, subdomain_id=1)
    define_formulation(subdomain)
    subdomain.u.name = f"u_{subdomain.id}"

# boundary conditions
bcs = []
for boundary_condition in list_of_bcs:
    volume_subdomain = surface_to_volume[boundary_condition.subdomain]
    bc = dolfinx.fem.Function(volume_subdomain.u.function_space)
    bc.x.array[:] = boundary_condition.value
    volume_subdomain.submesh.topology.create_connectivity(
        volume_subdomain.submesh.topology.dim - 1,
        volume_subdomain.submesh.topology.dim,
    )
    bc = dolfinx.fem.dirichletbc(
        bc,
        dolfinx.fem.locate_dofs_topological(
            volume_subdomain.u.function_space.sub(0),
            fdim,
            volume_subdomain.ft.find(boundary_condition.subdomain.id),
        ),
    )
    bcs.append(bc)

# Add coupling term to the interface
# Get interface markers on submesh b
for interface in list_of_interfaces:

    dInterface = ufl.Measure(
        "dS", domain=mesh, subdomain_data=mt, subdomain_id=interface
    )
    b_res = "+"
    t_res = "-"

    # look at the first facet on interface
    # and get the two cells that are connected to it
    # and get the material properties of these cells
    first_facet_interface = mt.find(interface)[0]
    c_plus, c_minus = (
        f_to_c.links(first_facet_interface)[0],
        f_to_c.links(first_facet_interface)[1],
    )
    id_minus, id_plus = ct.values[c_minus], ct.values[c_plus]

    for subdomain in list_of_interfaces[interface]:
        if subdomain.id == id_plus:
            subdomain_1 = subdomain
        if subdomain.id == id_minus:
            subdomain_2 = subdomain

    v_b = H.subdomain_to_test_function[subdomain_1](b_res)
    v_t = H.subdomain_to_test_function[subdomain_2](t_res)

    u_b = H.subdomain_to_solution[subdomain_1](b_res)
    u_t = H.subdomain_to_solution[subdomain_2](t_res)

    def mixed_term(u, v, n):
        return ufl.dot(ufl.grad(u), n) * v

    n = ufl.FacetNormal(mesh)
    n_b = n(b_res)
    n_t = n(t_res)
    cr = ufl.Circumradius(mesh)
    h_b = 2 * cr(b_res)
    h_t = 2 * cr(t_res)
    gamma = 400.0  # this needs to be "sufficiently large"

    K_b = K_S_fun(T(b_res), subdomain_1.material.K_S_0, subdomain_1.material.E_K_S)
    K_t = K_S_fun(T(t_res), subdomain_2.material.K_S_0, subdomain_2.material.E_K_S)

    F_0 = (
        -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
        - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface
    )

    F_1 = (
        +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
        - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface
    )
    F_0 += 2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_b * dInterface
    F_1 += -2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_t * dInterface

    subdomain_1.F += F_0
    subdomain_2.F += F_1

J = []
forms = []
for subdomain1 in list_of_subdomains:
    jac = []
    form = subdomain1.F
    for subdomain2 in list_of_subdomains:
        jac.append(
            dolfinx.fem.form(
                ufl.derivative(form, subdomain2.u), entity_maps=entity_maps
            )
        )
    J.append(jac)
    forms.append(dolfinx.fem.form(subdomain1.F, entity_maps=entity_maps))


solver = NewtonSolver(
    forms,
    J,
    [subdomain.u for subdomain in list_of_subdomains],
    bcs=bcs,
    max_iterations=10,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
solver.solve(1e-5)

for subdomain in list_of_subdomains:
    u_sub_0 = subdomain.u.sub(0).collapse()
    u_sub_0.name = "u_sub_0"

    u_sub_1 = subdomain.u.sub(1).collapse()
    u_sub_1.name = "u_sub_1"
    bp = dolfinx.io.VTXWriter(
        mesh.comm, f"u_{subdomain.id}.bp", [u_sub_0, u_sub_1], engine="BP4"
    )
    bp.write(0)
    bp.close()


# derived quantities
entity_maps[mesh] = bottom_domain.submesh_to_mesh

ds_b = ufl.Measure("ds", domain=bottom_domain.submesh, subdomain_data=bottom_domain.ft)
ds_t = ufl.Measure("ds", domain=top_domain.submesh, subdomain_data=top_domain.ft)
dx_b = ufl.Measure("dx", domain=bottom_domain.submesh)
dx = ufl.Measure("dx", domain=mesh)

n_b = ufl.FacetNormal(bottom_domain.submesh)
n_t = ufl.FacetNormal(top_domain.submesh)

form = dolfinx.fem.form(bottom_domain.u.sub(0) * dx_b)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(bottom_domain.u.sub(1) * dx_b)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(T * dx_b, entity_maps={mesh: bottom_domain.submesh_to_mesh})
print(dolfinx.fem.assemble_scalar(form))

id_interface = 5
form = dolfinx.fem.form(
    ufl.dot(
        D_fun(T, bottom_domain.material.D_0, bottom_domain.material.E_D)
        * ufl.grad(bottom_domain.u.sub(0)),
        n_b,
    )
    * ds_b(id_interface),
    entity_maps={mesh: bottom_domain.submesh_to_mesh},
)
print(dolfinx.fem.assemble_scalar(form))
form = dolfinx.fem.form(
    ufl.dot(
        D_fun(T, top_domain.material.D_0, top_domain.material.E_D)
        * ufl.grad(top_domain.u.sub(0)),
        n_t,
    )
    * ds_t(id_interface),
    entity_maps={mesh: top_domain.submesh_to_mesh},
)
print(dolfinx.fem.assemble_scalar(form))