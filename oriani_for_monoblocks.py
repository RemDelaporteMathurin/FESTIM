import festim as F
import fenics

my_model = F.Simulation()

import numpy as np

my_model.mesh = F.MeshFromVertices(np.linspace(0, 3e-3, num=2000))


tungsten = F.Material(
    id=1,
    D_0=4.1e-07,  # m2/s
    E_D=0.39,  # eV
    thermal_cond=173,  # W/mK
)

my_model.materials = tungsten


w_atom_density = 6.3e28  # atom/m3

trap_1 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=0.87,
    density=1.3e-3 * w_atom_density,
    materials=tungsten,
)
trap_2 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.2,
    density=4e-4 * w_atom_density,
    materials=tungsten,
)

my_model.traps = [trap_1, trap_2]

my_model.T = F.HeatTransferProblem(transient=False)

my_model.boundary_conditions = [
    F.ImplantationDirichlet(surfaces=1, phi=1e19, R_p=4.5e-9, D_0=4.1e-07, E_D=0.39),
    F.DirichletBC(surfaces=2, value=0, field=0),
    F.DirichletBC(surfaces=1, value=600, field="T"),
    F.DirichletBC(surfaces=2, value=600, field="T"),
]


# Stepsize
my_model.dt = F.Stepsize(
    initial_value=0.5,
    stepsize_change_ratio=1.1,
    dt_min=1e-05,
)

# Settings
my_model.settings = F.Settings(
    absolute_tolerance=1e8, relative_tolerance=1e-09, final_time=1e7
)

total_mobile = F.TotalVolume("solute", volume=1)
total_trap1 = F.TotalVolume("1", volume=1)
total_trap2 = F.TotalVolume("2", volume=1)

derived_quantities = F.DerivedQuantities(
    [total_mobile, total_trap1, total_trap2], show_units=True
)


my_model.exports = [derived_quantities]

my_model.initialise()
my_model.run()


# ---------- simplified model ------------

retention_simplified = []
t_simplified = []


def oriani_approx(c_m, trap, T):
    n = trap.density[0]
    k = trap.k_0 * fenics.exp(-trap.E_k / F.k_B / T)
    p = trap.p_0 * fenics.exp(-trap.E_p / F.k_B / T)
    return n / (1 + p / (k * c_m))


class SimplifiedModel(F.Simulation):
    def iterate(self):
        super().iterate()
        cm = self.h_transport_problem.mobile.mobile_concentration()
        c_t1 = oriani_approx(cm, trap_1, self.T.T)
        c_t2 = oriani_approx(cm, trap_2, self.T.T)
        retention = fenics.assemble((cm + c_t1 + c_t2) * fenics.dx)
        retention_simplified.append(retention)
        t_simplified.append(self.t)


simplified_model = SimplifiedModel()
simplified_model.mesh = my_model.mesh
simplified_model.materials = my_model.materials
simplified_model.T = my_model.T
simplified_model.boundary_conditions = my_model.boundary_conditions
simplified_model.dt = my_model.dt
simplified_model.settings = my_model.settings
simplified_model.exports = []
simplified_model.traps = []

simplified_model.initialise()
simplified_model.run()


import matplotlib.pyplot as plt

retention_full = (
    np.array(total_mobile.data)
    + np.array(total_trap1.data)
    + np.array(total_trap2.data)
)

plt.loglog(t_simplified, retention_simplified, label="Oriani approx", color="tab:red")
plt.loglog(total_mobile.t, retention_full, label="Full model", color="tab:green")
plt.legend()
plt.ylabel("Inventory (H/m2)")
plt.xlabel("Time (s)")
plt.show()
