import festim as F
import numpy as np
import time

assert hasattr(F, "Simulation"), "you should use FESTIM v1.x.x"


def run_festim_1(volume_file, facet_file):
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromXDMF(
        volume_file=volume_file,
        boundary_file=facet_file,
    )

    tungsten = F.Material(id=1, D_0=1, E_D=0, S_0=1, E_S=0)

    copper = F.Material(id=2, D_0=1, E_D=0, S_0=2, E_S=0)

    my_model.materials = [tungsten, copper]

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
                k_0=[1, 1],
                E_k=[tungsten.E_D, copper.E_D],
                p_0=[0.1, 0.1],
                E_p=[1.0, 0.5],
                density=[0.5, 0.5],
                materials=[tungsten, copper],
            ),
        ]
    )

    my_model.T = 600

    my_model.boundary_conditions = [
        F.DirichletBC(surfaces=[3], value=2, field=0),
        F.DirichletBC(surfaces=[5], value=0, field=0),
    ]

    my_model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=15,
        traps_element_type="DG",
        chemical_pot=True,
        transient=True,
        final_time=10,
        linear_solver="mumps",
    )

    my_model.dt = F.Stepsize(1)

    my_model.exports = F.Exports(
        [
            F.XDMFExport("solute", folder="results_festim_1", checkpoint=True),
            F.XDMFExport("1", folder="results_festim_1", checkpoint=True),
            F.XDMFExport("2", folder="results_festim_1", checkpoint=True),
        ]
    )
    my_model.initialise()
    my_model.run()


if __name__ == "__main__":
    times = []
    sizes = [0.025]  # , 0.05, 0.025]
    for size in sizes:
        print(f"Running for size {size}")
        volume_file = f"mesh/size_{size}/mesh.xdmf"
        facet_file = f"mesh/size_{size}/mf.xdmf"
        start_time = time.time()
        nb_cells = run_festim_1(volume_file=volume_file, facet_file=facet_file)
        end_time = time.time()
        ellapsed_time = end_time - start_time
        times.append(ellapsed_time)
    print(sizes)
    print(times)
