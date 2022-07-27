import FESTIM
from fenics import *
import sympy as sp
from pathlib import Path


def test_trap_density_xdmf_export_intergration_with_simultion(tmpdir):
    """Integration test for TrapDensityXDMF.write().
    Creates a FESTIM simulation and exports the trap density as an .xmdf file.
    An equivalent fenics function is created and is compared to that that read
    from the .xdmf file created. Ensures compatability with FESTIM.Simulation()
    """
    density_expr = 2 + FESTIM.x**2 + 2 * FESTIM.y

    my_model = FESTIM.Simulation(log_level=20)
    my_model.mesh = FESTIM.Mesh()
    my_model.mesh.mesh = UnitSquareMesh(30, 30)
    mat_1 = FESTIM.Material(D_0=1, E_D=0, id=1)
    my_model.materials = FESTIM.Materials([mat_1])
    trap_1 = FESTIM.Trap(
        k_0=1, E_k=0, p_0=1, E_p=0, density=density_expr, materials=mat_1
    )
    my_model.traps = FESTIM.Traps([trap_1])
    my_model.T = FESTIM.Temperature(value=300)
    my_model.settings = FESTIM.Settings(
        transient=False,
        absolute_tolerance=1e06,
        relative_tolerance=1e-08,
    )
    density_file = tmpdir.join("density1.xdmf")
    my_export = FESTIM.TrapDensityXDMF(
        trap=trap_1,
        label="density1",
        filename=str(Path(density_file)),
    )
    my_exports = FESTIM.Exports([my_export])
    my_model.exports = my_exports
    my_model.initialise()
    my_model.run()

    V = FunctionSpace(my_model.mesh.mesh, "CG", 1)

    density_expected = interpolate(FESTIM.as_expression(density_expr), V)

    density_read = Function(V)
    XDMFFile(str(Path(density_file))).read_checkpoint(density_read, "density1", -1)

    l2_error = errornorm(density_expected, density_read, "L2")
    assert l2_error < 2e-3


def test_trap_density_xdmf_export_write(tmpdir):
    """Test for TrapDensityXDMF.write()
    Creates a FESTIM density function and exports as an .xmdf file.
    An equivalent fenics function is created and is compared to that that read
    from the .xdmf file created.
    """
    # build
    mesh = UnitSquareMesh(30, 30)
    V = FunctionSpace(mesh, "CG", 1)
    V_vector = VectorFunctionSpace(mesh, "CG", 1, 2)
    density_expr = 2 + FESTIM.x + FESTIM.y
    expr = Expression(sp.printing.ccode(density_expr), degree=2)
    density_expected = interpolate(expr, V)

    density_file = tmpdir.join("density1.xdmf")

    trap_1 = FESTIM.Trap(1, 0, 1, 0, materials="1", density=density_expr)
    my_export = FESTIM.TrapDensityXDMF(
        trap=trap_1,
        label="density1",
        filename=str(Path(density_file)),
    )
    my_export.function = Function(V_vector).sub(1)

    # run
    my_export.write(t=1)

    # test
    density_read = Function(V)
    XDMFFile(str(Path(density_file))).read_checkpoint(density_read, "density1", -1)
    error_L2 = errornorm(density_expected, density_read, "L2")
    print(error_L2)
    assert error_L2 < 1e-10