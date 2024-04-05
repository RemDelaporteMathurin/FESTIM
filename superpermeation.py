import festim as F
import numpy as np
import matplotlib.pyplot as plt


model = F.Simulation()

vertices = np.concatenate(
    [
        np.linspace(0, 30e-9, num=200),
        np.linspace(30e-9, 3e-6, num=300),
        np.linspace(3e-6, 20e-6, num=200),
    ]
)
model.mesh = F.MeshFromVertices(vertices)

tungsten = F.Material(
    id=1,
    D_0=2.9e-8,  # m2/s
    E_D=0.04,  # eV
)

model.materials = tungsten


implantation_time = 400  # s

ion_flux = 2.5e19

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=10e-9, width=2.5e-9, volume=1  # H/m2/s  # m  # m
)

model.sources = [source_term]

model.T = F.Temperature(500)

model.boundary_conditions = [
    F.RecombinationFlux(Kr_0=1e-15, E_Kr=2.5, order=2, surfaces=1),
    F.DirichletBC(value=0, surfaces=2, field=0),
]

model.dt = F.Stepsize(initial_value=1e-5, stepsize_change_ratio=1.1)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-09,
    final_time=1,
)

permeation_flux = F.HydrogenFlux(surface=2)
retro_desorbed_flux = F.HydrogenFlux(surface=1)

derived_quantities = F.DerivedQuantities(
    [
        retro_desorbed_flux,
        permeation_flux,
    ]
)

model.exports = [derived_quantities]

permeated_fluxes = []
retro_desorbed_fluxes = []

E_Kr_values = np.linspace(0.1, 2, num=20)
for E_Kr in E_Kr_values:

    model.boundary_conditions[0].E_Kr = E_Kr
    permeation_flux.data = []
    retro_desorbed_flux.data = []
    permeation_flux.t = []
    retro_desorbed_flux.t = []
    model.initialise()
    model.run()

    permeated_fluxes.append(abs(permeation_flux.data[-1]))
    retro_desorbed_fluxes.append(abs(retro_desorbed_flux.data[-1]))

plt.stackplot(
    E_Kr_values,
    permeated_fluxes,
    retro_desorbed_fluxes,
    labels=["Permeated", "Retro-desorbed"],
)
plt.xlabel("Recombination activation energy (eV)")
plt.ylabel("Outgassing flux (H/m2/s)")
plt.annotate(
    "Permeated", (1.5, 1e19), color="white", weight="bold", ha="center", fontsize=16
)
plt.annotate(
    "Retro-desorbed",
    (0.6, 1e19),
    color="white",
    weight="bold",
    ha="center",
    fontsize=16,
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.savefig("superpermeation.svg")
plt.show()
