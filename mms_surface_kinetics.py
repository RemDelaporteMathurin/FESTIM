import festim as F
import matplotlib.pyplot as plt
import numpy as np

# Create the FESTIM model
my_model = F.Simulation()

my_model.mesh = F.MeshFromVertices(np.linspace(0, 1, num=100))

# Variational formulation
exact_solution_cm = lambda x, t: 1 + 2 * x**2 + x + 2 * t
exact_solution_cs = lambda t: 3 * t

my_model.initial_conditions = [
    F.InitialCondition(field="solute", value=exact_solution_cm(x=F.x, t=F.t))
]

D = 20

my_model.sources = [
    F.Source(2 - 4 * D, volume=1, field="solute"),
]

lambda_IS = 0.002
J_vs = 3 - D - 2 * lambda_IS
surface_kinetics = F.SurfaceKinetics(
    k_sb=2,
    k_bs=2,
    lambda_IS=lambda_IS,
    n_surf=3,
    n_IS=4,
    J_vs=J_vs,
    surfaces=[1],
    initial_condition=0,
)

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[2], value=exact_solution_cm(x=F.x, t=F.t), field="solute"),
    surface_kinetics,
]

my_model.materials = F.Material(id=1, D_0=D, E_D=0)

my_model.T = F.Temperature(500)  # ignored in this problem

my_model.settings = F.Settings(
    absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=3
)

my_model.dt = F.Stepsize(
    initial_value=1e-2,
)
export_times = [1, 2, 3]
my_model.exports = [
    # F.XDMFExport("solute", checkpoint=False),
    F.TXTExport("solute", filename="./mobile_conc.txt", times=export_times),
    F.DerivedQuantities([F.AdsorbedHydrogen(surface=1)]),
]
my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

data = np.genfromtxt("mobile_conc.txt", names=True, delimiter=",")
for t in export_times:
    x = data["x"]
    y = data[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    (l1,) = plt.plot(
        x,
        exact_solution_cm(np.array(x), t),
        "--",
        label=f"exact t={t}",
    )
    plt.scatter(
        x[::10],
        y[::10],
        label=f"t={t}",
        color=l1.get_color(),
    )

plt.legend(reverse=True)

plt.figure()
c_s_computed = my_model.exports[1][0].data
t = my_model.exports[1][0].t

plt.plot(t, c_s_computed, label="computed")
plt.plot(t, exact_solution_cs(np.array(t)), label="exact")
plt.ylabel("$c_s$")
plt.xlabel("t")
plt.legend()
plt.show()
