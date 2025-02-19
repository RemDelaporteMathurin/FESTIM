import festim as F
import numpy as np
import matplotlib.pyplot as plt

import dolfinx.fem as fem
import ufl
from dolfinx.log import set_log_level, LogLevel

# set_log_level(LogLevel.INFO)


class FluxFromSurfaceReaction(F.SurfaceFlux):
    def __init__(self, reaction: F.SurfaceReactionBC):
        super().__init__(
            F.Species(),  # just a dummy species here
            reaction.subdomain,
        )
        self.reaction = reaction.flux_bcs[0]

    def compute(self, ds):
        self.value = fem.assemble_scalar(
            fem.form(self.reaction.value_fenics * ds(self.surface.id))
        )
        self.data.append(self.value)


pd_thickness = 0.025e-3  # m
upstream_h2_pressure = 0.063  # Pa
upstream_d2_pressure = 0.1  # Pa
temperature = 870  # K

import h_transport_materials as htm

pd_recomb_coeff = htm.recombination_coeffs.filter(material=htm.PALLADIUM).mean()
pd_diss_coeff = htm.dissociation_coeffs.filter(material=htm.PalladiumAlloy).mean()
pd_diffusion_coeff = htm.diffusivities.filter(material=htm.PALLADIUM).mean()

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(vertices=np.linspace(0, pd_thickness, 300))
my_mat = F.Material(
    name="Pd",
    D_0=pd_diffusion_coeff.pre_exp.magnitude,
    E_D=pd_diffusion_coeff.act_energy.magnitude,
)
vol = F.VolumeSubdomain1D(id=1, borders=[0, pd_thickness], material=my_mat)
left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=pd_thickness)

my_model.subdomains = [vol, left, right]

H = F.Species("H")
D = F.Species("D")
my_model.species = [H, D]

my_model.temperature = temperature

surface_reaction_hh_left = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=upstream_h2_pressure,
    k_r0=pd_recomb_coeff.pre_exp.magnitude,
    E_kr=pd_recomb_coeff.act_energy.magnitude,
    k_d0=pd_diss_coeff.pre_exp.magnitude * 1.23,
    E_kd=pd_diss_coeff.act_energy.magnitude,
    subdomain=left,
)

surface_reaction_dd_left = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=upstream_d2_pressure,
    k_r0=pd_recomb_coeff.pre_exp.magnitude / 1.23 / 2,
    E_kr=pd_recomb_coeff.act_energy.magnitude,
    k_d0=pd_diss_coeff.pre_exp.magnitude,
    E_kd=pd_diss_coeff.act_energy.magnitude,
    subdomain=left,
)

surface_reaction_hd_right = F.SurfaceReactionBC(
    reactant=[H, D],
    gas_pressure=0,
    k_r0=pd_recomb_coeff.pre_exp.magnitude,
    E_kr=pd_recomb_coeff.act_energy.magnitude,
    k_d0=pd_diss_coeff.pre_exp.magnitude,
    E_kd=pd_diss_coeff.act_energy.magnitude,
    subdomain=right,
)

surface_reaction_hh_right = F.SurfaceReactionBC(
    reactant=[H, H],
    gas_pressure=0,
    k_r0=pd_recomb_coeff.pre_exp.magnitude,
    E_kr=pd_recomb_coeff.act_energy.magnitude,
    k_d0=pd_diss_coeff.pre_exp.magnitude,
    E_kd=pd_diss_coeff.act_energy.magnitude,
    subdomain=right,
)

surface_reaction_dd_right = F.SurfaceReactionBC(
    reactant=[D, D],
    gas_pressure=0,
    k_r0=pd_recomb_coeff.pre_exp.magnitude / 1.23 / 2,
    E_kr=pd_recomb_coeff.act_energy.magnitude,
    k_d0=pd_diss_coeff.pre_exp.magnitude,
    E_kd=pd_diss_coeff.act_energy.magnitude,
    subdomain=right,
)

my_model.boundary_conditions = [
    surface_reaction_hh_left,
    surface_reaction_dd_left,
    surface_reaction_hd_right,
    surface_reaction_hh_right,
    surface_reaction_dd_right,
]

H_flux_right = F.SurfaceFlux(H, right)
H_flux_left = F.SurfaceFlux(H, left)
D_flux_right = F.SurfaceFlux(D, right)
D_flux_left = F.SurfaceFlux(D, left)
HD_flux = FluxFromSurfaceReaction(surface_reaction_hd_right)
HH_flux = FluxFromSurfaceReaction(surface_reaction_hh_right)
DD_flux = FluxFromSurfaceReaction(surface_reaction_dd_right)


my_model.exports = [
    H_flux_left,
    H_flux_right,
    D_flux_left,
    D_flux_right,
    HD_flux,
    HH_flux,
    DD_flux,
]


my_model.settings = F.Settings(atol=1e11, rtol=1e-10, final_time=10, transient=True)

my_model.settings.stepsize = 0.2

all_d_desorption_fluxes = []
hh_desorption_fluxes = []
hd_desorption_fluxes = []
dd_desorption_fluxes = []
upstream_d_pressures = np.geomspace(4e-3, 1, num=5)

for upstream_d_pressure in upstream_d_pressures:
    for flux_bc in surface_reaction_dd_left.flux_bcs:
        flux_bc.gas_pressure = upstream_d_pressure

    my_model.initialise()
    my_model.run()

    # ------ Post processsing ------ #

    # convert all data to mol
    for export in my_model.exports:
        avogadro = 6.022e23
        export.data = np.array(export.data) / avogadro

    all_d_desorption_fluxes.append(np.abs(D_flux_right.data)[-1])
    print(
        f"Desorption flux at {upstream_d_pressure} Pa: {all_d_desorption_fluxes[-1]} mol/m^2/s"
    )

    hh_desorption_fluxes.append(np.abs(HH_flux.data)[-1])
    hd_desorption_fluxes.append(np.abs(HD_flux.data)[-1])
    dd_desorption_fluxes.append(np.abs(DD_flux.data)[-1])


import pandas as pd

# read experimental data
exp_data = pd.read_csv(
    "co_permeation_exp_data.csv",
    names=["H2_X", "H2_Y", "D2_X", "D2_Y", "HD_X", "HD_Y"],
    skiprows=2,
)

from pypalettes import load_cmap

cmap = load_cmap("Acadia")

plt.scatter(exp_data["H2_X"], exp_data["H2_Y"], marker="o", label="H2", color=cmap(0))
plt.scatter(exp_data["D2_X"], exp_data["D2_Y"], marker="^", label="D2", color=cmap(1))
plt.scatter(exp_data["HD_X"], exp_data["HD_Y"], marker="s", label="HD", color=cmap(2))

plt.plot(upstream_d_pressures, hh_desorption_fluxes, label="HH", color=cmap(0))
plt.plot(upstream_d_pressures, dd_desorption_fluxes, label="DD", color=cmap(1))
plt.plot(upstream_d_pressures, hd_desorption_fluxes, label="HD", color=cmap(2))

plt.xlabel("Upstream D pressure (Pa)")
plt.ylabel("Desorption flux (mol/m^2/s)")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-8, 1e-3)
plt.legend()
plt.savefig("co_permeation_desorption_flux.png")

plt.figure()
plt.stackplot(
    H_flux_left.t,
    np.abs(H_flux_left.data),
    np.abs(D_flux_left.data),
    labels=["H_in", "D_in"],
)
plt.stackplot(
    H_flux_right.t,
    -np.abs(H_flux_right.data),
    -np.abs(D_flux_right.data),
    labels=["H_out", "D_out"],
)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Flux (mol/m^2/s)")
plt.savefig("co_permeation_in_out.png")

plt.figure()
plt.stackplot(
    HD_flux.t,
    np.abs(HH_flux.data),
    np.abs(HD_flux.data),
    np.abs(DD_flux.data),
    labels=["HH", "HD", "DD"],
)
plt.legend(reverse=True)
plt.xlabel("Time (s)")
plt.ylabel("Flux (mol (molecule)/m^2/s)")


plt.figure()
plt.plot(H_flux_right.t, -np.array(H_flux_right.data), label="from gradient (H)")
plt.plot(
    H_flux_right.t,
    2 * np.array(HH_flux.data) + np.array(HD_flux.data),
    linestyle="--",
    label="from reaction rates (H)",
)

plt.plot(D_flux_right.t, -np.array(D_flux_right.data), label="from gradient (D)")
plt.plot(
    D_flux_right.t,
    2 * np.array(DD_flux.data) + np.array(HD_flux.data),
    linestyle="--",
    label="from reaction rates (D)",
)
plt.xlabel("Time (s)")
plt.ylabel("Flux (mol/m^2/s)")
plt.legend()
plt.savefig("co_permeation.png")
plt.show()

# # check that H_flux_right == 2*HH_flux + HD_flux
# H_flux_from_gradient = -np.array(H_flux_right.data)
# H_flux_from_reac = 2 * np.array(HH_flux.data) + np.array(HD_flux.data)
# assert np.allclose(
#     H_flux_from_gradient,
#     H_flux_from_reac,
#     rtol=0.5e-2,
#     atol=0.005,
# )
# # check that D_flux_right == 2*DD_flux + HD_flux
# D_flux_from_gradient = -np.array(D_flux_right.data)
# D_flux_from_reac = 2 * np.array(DD_flux.data) + np.array(HD_flux.data)
# assert np.allclose(
#     D_flux_from_gradient,
#     D_flux_from_reac,
#     rtol=0.5e-2,
#     atol=0.005,
# )
