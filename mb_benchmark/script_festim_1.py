import festim as F
import numpy as np

assert hasattr(F, "Simulation"), "you should use FESTIM v1.x.x"

my_model = F.Simulation()
my_model.mesh = F.MeshFromXDMF(
    volume_file="mesh/mesh.xdmf",
    boundary_file="mesh/mf.xdmf",
)

tungsten = F.Material(
    id=1,
    D_0=1,
    E_D=0.390,
    S_0=1,
    E_S=1.04,
)

copper = F.Material(
    id=2,
    D_0=1,
    E_D=0.387,
    S_0=2,
    E_S=0.572,
)

cucrzr = F.Material(
    id=3,
    D_0=1,
    E_D=0.418,
    S_0=3,
    E_S=0.387,
)

my_model.materials = [tungsten, copper, cucrzr]

my_model.traps = F.Traps(
    [
        F.Trap(
            k_0=1,
            E_k=tungsten.E_D,
            p_0=0.1,
            E_p=0.87,
            density=0.5,
            materials=tungsten,
        ),
        F.Trap(
            k_0=[1, 1, 1],
            E_k=[tungsten.E_D, copper.E_D, cucrzr.E_D],
            p_0=[0.1, 0.1, 0.1],
            E_p=[1.0, 0.5, 0.85],
            density=[0.5, 0.5, 0.5],
            materials=[tungsten, copper, cucrzr],
        ),
    ]
)

my_model.T = 600

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[4], value=2, field=0),
    F.DirichletBC(surfaces=[5], value=0, field=0),
]

my_model.settings = F.Settings(
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
    maximum_iterations=15,
    traps_element_type="DG",
    chemical_pot=True,
    transient=True,
    final_time=1000,
    linear_solver="mumps",
)

my_model.dt = F.Stepsize(100)

my_model.exports = F.Exports(
    [
        F.XDMFExport("T", folder="results_festim_1"),
        F.XDMFExport("solute", folder="results_festim_1", checkpoint=True),
        F.XDMFExport("retention", folder="results_festim_1", checkpoint=True),
        F.XDMFExport("1", folder="results_festim_1", checkpoint=True),
        F.XDMFExport("2", folder="results_festim_1", checkpoint=True),
    ]
)

if __name__ == "__main__":
    my_model.initialise()
    my_model.run()
