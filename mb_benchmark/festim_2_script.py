import festim as F
import numpy as np

assert hasattr(
    F, "HydrogenTransportProblem"
), "you should use FESTIM on the fenicsx branch"


import dolfinx

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

my_model = F.HTransportProblemDiscontinuous()

id_W = 6  # volume W
id_Cu = 7  # volume Cu
id_CuCrZr = 8  # volume CuCrZr
id_W_top = 9
id_coolant = 10
id_poloidal_gap_W = 11
id_poloidal_gap_Cu = 12
id_toroidal_gap = 13
id_bottom = 14
id_top_pipe = 15
id_interface_W_Cu = 16
id_interface_Cu_cucrzr = 17

my_model.mesh = F.MeshFromXDMF(
    volume_file="mesh/mesh_cells.xdmf",
    facet_file="mesh/mesh_facets.xdmf",
)

ct = my_model.mesh.define_volume_meshtags()
ft = my_model.mesh.define_surface_meshtags()
print(f"Loaded mesh with {len(ct.values)} cells and {len(ft.values)} facets")


def polynomial(coeffs, x, main):
    val = coeffs[0]
    for i in range(1, 4):
        if main:
            val += coeffs[i] * np.float_power(x, i)
        else:
            val += coeffs[i] * x**i
    return val


def thermal_cond_W(T, main=False):
    coeffs = [1.75214e2, -1.07335e-1, 5.03006e-5, -7.84154e-9]
    return polynomial(coeffs, T, main=main)


def thermal_cond_Cu(T, main=False):
    coeffs = [4.02301e02, -7.88669e-02, 3.76147e-05, -3.93153e-08]
    return polynomial(coeffs, T, main=main)


def thermal_cond_CuCrZr(T, main=False):
    coeffs = [3.12969e2, 2.57678e-01, -6.45110e-4, 5.25780e-7]
    return polynomial(coeffs, T, main=main)


atom_density_W = 6.3222e28  # atomic density m^-3
atom_density_Cu = 8.4912e28  # atomic density m^-3
atom_density_CuCrZr = 2.6096e28  # atomic density m^-3

tungsten = F.Material(
    D_0=4.10e-7,
    E_D=0.390,
    K_S_0=1.870e24,
    E_K_S=1.04,
    thermal_conductivity=thermal_cond_W,
)

copper = F.Material(
    D_0=6.60e-7,
    E_D=0.387,
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=thermal_cond_Cu,
)

cucrzr = F.Material(
    D_0=3.92e-7,
    E_D=0.418,
    K_S_0=4.28e23,
    E_K_S=0.387,
    thermal_conductivity=thermal_cond_CuCrZr,
)

tungsten_vol = F.VolumeSubdomain(id=id_W, material=tungsten)
cucrzr_vol = F.VolumeSubdomain(id=id_CuCrZr, material=cucrzr)
cu_vol = F.VolumeSubdomain(id=id_Cu, material=copper)
W_top = F.SurfaceSubdomain(id=id_W_top)
coolant = F.SurfaceSubdomain(id=id_coolant)
poloidal_gap_W = F.SurfaceSubdomain(id=id_poloidal_gap_W)
poloidal_gap_Cu = F.SurfaceSubdomain(id=id_poloidal_gap_Cu)
toroidal_gap = F.SurfaceSubdomain(id=id_toroidal_gap)
bottom = F.SurfaceSubdomain(id=id_bottom)
top_pipe = F.SurfaceSubdomain(id=id_top_pipe)


my_model.subdomains = [
    tungsten_vol,
    cucrzr_vol,
    cu_vol,
    W_top,
    coolant,
    poloidal_gap_W,
    poloidal_gap_Cu,
    toroidal_gap,
    bottom,
    top_pipe,
]

H = F.Species(name="H", subdomains=my_model.volume_subdomains)
H_trapped_W1 = F.Species(name="H_trapped_W1", subdomains=[tungsten_vol])
H_trapped_W2 = F.Species(name="H_trapped_W2", subdomains=[tungsten_vol])
H_trapped_cu = F.Species(name="H_trapped_cu", subdomains=[cu_vol])
H_trapped_cucrzr = F.Species(name="H_trapped_cucrzr", subdomains=[cucrzr_vol])
my_model.species = [H, H_trapped_W1, H_trapped_W2, H_trapped_cu, H_trapped_cucrzr]

empty_trap_W1 = F.ImplicitSpecies(n=1.3e-3 * atom_density_W, others=[H_trapped_W1])
empty_trap_W2 = F.ImplicitSpecies(n=4e-4 * atom_density_W, others=[H_trapped_W2])
empty_trap_cu = F.ImplicitSpecies(n=5e-5 * atom_density_Cu, others=[H_trapped_cu])
empty_trap_cucrzr = F.ImplicitSpecies(
    n=5e-5 * atom_density_CuCrZr, others=[H_trapped_cucrzr]
)

my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_trap_W1],
        product=[H_trapped_W1],
        k_0=8.96e-17,
        E_k=tungsten.E_D,
        E_p=0.87,
        p_0=1e13,
        volume=tungsten_vol,
    ),
    F.Reaction(
        reactant=[H, empty_trap_W2],
        product=[H_trapped_W2],
        k_0=8.96e-17,
        E_k=tungsten.E_D,
        E_p=1.0,
        p_0=1e13,
        volume=tungsten_vol,
    ),
    F.Reaction(
        reactant=[H, empty_trap_cu],
        product=[H_trapped_cu],
        k_0=6e-17,
        E_k=copper.E_D,
        E_p=0.5,
        p_0=8e13,
        volume=cu_vol,
    ),
    F.Reaction(
        reactant=[H, empty_trap_cucrzr],
        product=[H_trapped_cucrzr],
        k_0=1.2e-16,
        E_k=cucrzr.E_D,
        E_p=0.85,
        p_0=8e13,
        volume=cucrzr_vol,
    ),
]

my_heat_transfer_model = F.HeatTransferProblem()
my_heat_transfer_model.mesh = my_model.mesh
my_heat_transfer_model.boundary_conditions = [
    F.HeatFluxBC(subdomain=W_top, value=10e6),
    F.FixedTemperatureBC(subdomain=coolant, value=323),
]

my_heat_transfer_model.subdomains = my_model.subdomains

my_heat_transfer_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_heat_transfer_model.exports = [
    F.VTXExportForTemperature(filename="results/fenicsx/temperature.bp"),
]

my_heat_transfer_model.initialise()
my_heat_transfer_model.run()

my_model.temperature = my_heat_transfer_model.u
my_model.settings = F.Settings(atol=None, rtol=None)
my_model.settings.stepsize = F.Stepsize(
    initial_value=1e5, growth_factor=1.1, target_nb_iterations=4
)


# TODO need to remesh the geometry in SALOME in order to tag the interfaces

my_model.interfaces = [
    F.Interface(
        parent_mesh=None,
        mt=None,
        id=id_interface_W_Cu,
        subdomains=[tungsten_vol, cu_vol],
    ),
    F.Interface(
        parent_mesh=None,
        mt=None,
        id=id_interface_Cu_cucrzr,
        subdomains=[cu_vol, cucrzr_vol],
    ),
]


my_model.surface_to_volume = {
    W_top: tungsten_vol,
    coolant: cucrzr_vol,
    poloidal_gap_W: tungsten_vol,
    poloidal_gap_Cu: cu_vol,
    toroidal_gap: tungsten_vol,
    bottom: tungsten_vol,
    top_pipe: cucrzr_vol,
}

# my_model.initialise()
# my_model.run()
